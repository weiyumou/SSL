import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import utils
from data import SSLTrainDataset, SSLValDataset, fast_collate
from ranger import Ranger
from train import apex_train, apex_validate
from utils import save_checkpoint


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
    parser.add_argument('--do_ssl',
                        help='Whether to do SSL',
                        action="store_true")
    parser.add_argument('--download',
                        help='Whether to download datasets',
                        action="store_true")
    return parser.parse_args()


best_acc = 0
use_apex = True
mean = (0.4226, 0.4120, 0.3636)
std = (0.2615, 0.2545, 0.2571)
scale = 255

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    use_apex = False


def main():
    global best_acc, use_apex, mean, std, scale

    args = parse_args()
    args.mean, args.std, args.scale, args.use_apex = mean, std, scale, use_apex

    if args.use_apex:
        print("opt_level = {}".format(args.opt_level))
        print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
        print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
        print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    if args.deterministic:
        cudnn.deterministic = True
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1 and args.use_apex

    print(f"Use Apex: {args.use_apex}")
    print(f"Distributed Training Enabled: {args.distributed}")
    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        # Scale learning rate based on global batch size
        args.lr *= args.batch_size * args.world_size / 256

    if args.use_apex:
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    model = models.ResNet18(args.num_patches, args.num_angles)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()
    optimiser = Ranger(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().cuda()

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.use_apex:
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
        model = DDP(model, delay_allreduce=True)
    else:
        model = nn.DataParallel(model)

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

    if args.do_ssl:
        stl_unlabeled = datasets.STL10(root=args.data, split='unlabeled', download=args.download)
        indices = list(range(len(stl_unlabeled)))
        train_indices = indices[:int(len(indices) * 0.9)]
        val_indices = indices[int(len(indices) * 0.9):]
        train_dataset = SSLTrainDataset(Subset(stl_unlabeled, train_indices), args.num_patches, args.num_angles,
                                        args.poisson_rate)
        val_dataset = SSLValDataset(Subset(stl_unlabeled, val_indices), args.num_patches, args.num_angles)

        train_sampler = None
        val_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                  num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                  collate_fn=fast_collate)

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=fast_collate)

        if args.evaluate:
            val_loss, val_acc = apex_validate(val_loader, model, criterion, args)
            utils.logger.info(f"Val Loss = {val_loss}, Val Accuracy = {val_acc}")
            return

        # Create dir to save model and command-line args
        model_dir = time.ctime().replace(" ", "_").replace(":", "_")
        model_dir = os.path.join("models", model_dir)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

        writer = SummaryWriter()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            train_loss, train_acc = apex_train(train_loader, model, criterion, optimiser, args)

            # evaluate on validation set
            val_loss, val_acc = apex_validate(val_loader, model, criterion, args)

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
                }, is_best, model_dir)

                utils.logger.info(f"Epoch {epoch}: Train Loss = {train_loss}, Train Accuracy = {train_acc}")
                utils.logger.info(f"Epoch {epoch}: Val Loss = {val_loss}, Val Accuracy = {val_acc}")
                writer.add_scalars("Loss", {"train_loss": train_loss, "val_loss": val_loss}, epoch)
                writer.add_scalars("Accuracy", {"train_acc": train_acc, "val_acc": val_acc}, epoch)
                writer.add_scalar("Poisson_Rate", train_loader.dataset.pdist.rate, epoch)

        writer.close()


if __name__ == '__main__':
    main()
