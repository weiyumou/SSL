import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import models
from data import SSLTrainDataset, SSLValDataset, random_rotate
from ranger import Ranger

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR',
                        help='Initial learning rate. Will be scaled by <global batch size>/256: '
                             'args.lr = args.lr*float(args.batch_size*args.world_size)/256.  '
                             'A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--num_angles',
                        type=int,
                        help='Number of rotation angles',
                        default=4)
    parser.add_argument('--num_patches',
                        type=int,
                        help='Number of patches to extract from an image',
                        default=9)
    parser.add_argument('--learn_prd',
                        type=int,
                        help='Number of epochs before providing harder examples',
                        default=10)
    parser.add_argument('--poisson_rate',
                        type=int,
                        help='The initial poisson rate lambda',
                        default=2)
    parser.add_argument('--download',
                        help='Whether to download datasets',
                        action="store_true")
    return parser.parse_args()


cudnn.benchmark = True


def fast_collate(batch):
    images, rotations, perms = list(zip(*batch))
    img_tensors = []
    for img in images:
        nump_array = np.asarray(img, dtype=np.float)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        img_tensors.append(torch.from_numpy(nump_array))

    images = torch.stack(img_tensors, dim=0)
    rotations = torch.stack(rotations, dim=0)
    perms = torch.stack(perms, dim=0)
    return images, rotations, perms


best_acc = 0


def main():
    global best_acc

    args = parse_args()

    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    if args.deterministic:
        cudnn.deterministic = True
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    print(f"Distributed Training Enabled: {args.distributed}")
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    model = models.ResNet18(args.num_patches, args.num_angles)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()

    # Scale learning rate based on global batch size
    args.lr *= args.batch_size * args.world_size / 256
    # optimiser = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimiser = Ranger(model.parameters(), lr=args.lr)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimiser = amp.initialize(model, optimiser,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            global best_acc
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_acc = checkpoint['best_acc']
                args.poisson_rate = checkpoint["poisson_rate"]
                model.load_state_dict(checkpoint['state_dict'])
                optimiser.load_state_dict(checkpoint['optimiser'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # Data loading code
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    crop_size = 225
    val_size = 256

    imagenet_train = datasets.ImageFolder(root=train_dir,
                                          transform=transforms.Compose([
                                              transforms.RandomResizedCrop(crop_size),
                                          ]))
    train_dataset = SSLTrainDataset(imagenet_train, args.num_patches, args.num_angles)
    imagenet_val = datasets.ImageFolder(root=val_dir,
                                        transform=transforms.Compose([
                                            transforms.Resize(val_size),
                                            transforms.CenterCrop(crop_size),
                                        ]))
    val_dataset = SSLValDataset(imagenet_val, args.num_patches, args.num_angles)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=fast_collate)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    writer = SummaryWriter()
    train_loader.dataset.set_poisson_rate(args.poisson_rate)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimiser, epoch, args)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, args)

        if (epoch + 1) % args.learn_prd == 0:
            args.poisson_rate += 1
            train_loader.dataset.set_poisson_rate(args.poisson_rate)

        # remember best Acc and save checkpoint
        if args.local_rank == 0:
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimiser': optimiser.state_dict(),
                "poisson_rate": args.poisson_rate
            }, is_best)

            writer.add_scalars("Loss", {"train_loss": train_loss, "val_loss": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train_acc": train_acc, "val_acc": val_acc}, epoch)
            writer.add_scalar("Poisson_Rate", train_loader.dataset.pdist.rate, epoch)

    writer.close()


class DataPrefetcher():
    def __init__(self, loader, num_patches):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().reshape(1, 3, 1, 1)
        self.num_patches = num_patches
        self.preload()

    def preload(self):
        try:
            next_input, next_rotation, next_perm = next(self.loader)
            with torch.no_grad():
                self.next_input, self.next_label = random_rotate(next_input, self.num_patches, next_rotation, next_perm)
        except StopIteration:
            self.next_input = None
            self.next_label = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        labels = self.next_label
        if inputs is not None:
            inputs.record_stream(torch.cuda.current_stream())
        if labels is not None:
            labels.record_stream(torch.cuda.current_stream())
        self.preload()
        return inputs, labels


def train(train_loader, model, criterion, optimiser, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    prefetcher = DataPrefetcher(train_loader, args.num_patches)
    inputs, labels = prefetcher.next()
    i = 0
    while inputs is not None:
        i += 1

        outputs = model(inputs)
        outputs = outputs.reshape(-1, args.num_angles)
        loss = criterion(outputs, labels)

        optimiser.zero_grad()
        with amp.scale_loss(loss, optimiser) as scaled_loss:
            scaled_loss.backward()
        optimiser.step()

        inputs, labels = prefetcher.next()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            preds = torch.argmax(outputs, dim=1)
            corrects = torch.sum(preds == labels).item()
            acc = corrects / labels.numel()

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), labels.numel())
            top1.update(acc, labels.numel())
            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, i, len(train_loader),
                    args.world_size * args.batch_size / batch_time.val,
                    args.world_size * args.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses, top1=top1))

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()

    prefetcher = DataPrefetcher(val_loader, args.num_patches)
    inputs, labels = prefetcher.next()
    i = 0
    while inputs is not None:
        i += 1
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.reshape(-1, args.num_angles)
            loss = criterion(outputs, labels)

        # measure accuracy and record loss
        preds = torch.argmax(outputs, dim=1)
        corrects = torch.sum(preds == labels).item()
        acc = corrects / labels.numel()

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), labels.numel())
        top1.update(acc, labels.numel())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        inputs, labels = prefetcher.next()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                i, len(val_loader),
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Acc {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
