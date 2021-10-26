'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
from socket import AddressFamily
from imagenet import validate
import os
import random
import shutil
from threading import local
import time
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import models as customized_models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dataloaders import *
from tensorboardX import SummaryWriter
from Rog_utils import Local_Worker, Parameter_Server
import multiprocessing as mp
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch',
                    choices=DATA_BACKEND_CHOICES)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:13456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')
parser.add_argument('--threshold', default=2, type=int, metavar='threshold',
                    help='staleness threshold')
parser.add_argument('--fix', default=0.0, type=float, metavar='fix',
                    help='fix time')
best_prec1 = 0
ps_ip = "localhost"
ps_port = 12333
MTU=1500
def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        print("initializing process group")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
    print("=> creating model '{}'".format(args.arch))
    print(args.arch)
    # model = models.__dict__[args.arch](width_mult=args.width_mult)
    model = models.__dict__[args.arch]()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    if args.rank==0:
        # create model
        

        # if not args.distributed:
        #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #         model.features = torch.nn.DataParallel(model.features)
        #         # model.cuda()
        #     else:
        #         model = torch.nn.DataParallel(model)
        #         # model = torch.nn.DataParallel(model).cuda()
        # else:
        #     # model.cuda()
        #     model = torch.nn.parallel.DistributedDataParallel(model)

        # define loss function (criterion) and optimizer
        #criterion = nn.CrossEntropyLoss().cuda()

       
        
        # optionally resume from a checkpoint
        title = 'ImageNet-' + args.arch
        if not os.path.isdir(args.checkpoint):
            mkdir_p(args.checkpoint)

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
                args.checkpoint = os.path.dirname(args.resume)
                # logger = Logger(os.path.join(args.checkpoint, "threshold="+ str(args.threshold) + '_log.txt'), title=title, resume=True)
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        # else:
        # logger = Logger(os.path.join(args.checkpoint, "threshold="+ str(args.threshold) + '_log.txt'), title=title)
        # logger.set_names(
        #     ['Epoch', 'Timestamp', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Top-1 Acc.', 'Valid Top-1 Acc.',
        #     'Train Top-5 Acc.', 'Valid Top 5 Acc.'])


    cudnn.benchmark = True

    # Data loading code
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()

    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)
    if args.rank==0:
        if args.evaluate:
            from collections import OrderedDict
            if os.path.isfile(args.weight):
                print("=> loading pretrained weight '{}'".format(args.weight))
                source_state = torch.load(args.weight)
                target_state = OrderedDict()
                for k, v in source_state.items():
                    if k[:7] != 'module.':
                        k = 'module.' + k
                    target_state[k] = v
                model.load_state_dict(target_state)
            else:
                print("=> no weight found at '{}'".format(args.weight))

            return

    # visualization
    # writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))
    communication_library="rog"
    if args.rank==0:
        parameter_server=Parameter_Server(ps_ip,ps_port,args.world_size-1,args.threshold,model,optimizer,communication_library,MTU)
    else:
        criterion = nn.CrossEntropyLoss()
        local_worker=Local_Worker(args,model,ps_ip,ps_port,train_loader, train_loader_len,val_loader, val_loader_len ,criterion,optimizer,communication_library)

    if args.rank!=0:
        start_time = time.time()
        print("epoch\ttime\ttrain_loss\ttrain_top1\ttrain_top5")
        for epoch in range(args.start_epoch, args.epochs):
            # if args.distributed:
            #     train_sampler.set_epoch(epoch)
            print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

            # train for one epoch
            train_loss, train_top1_acc, train_top5_acc, time_elapsed = local_worker.train(epoch, start_time)

            # evaluate on validation set
            # val_loss, val_top1_acc, val_top5_acc, lr = local_worker.validate(start_time, epoch)

            # append logger file
            # logger.append([epoch + 1, time_elapsed.total_seconds(), lr, train_loss, val_loss, train_top1_acc, val_top1_acc,
            #             train_top5_acc, val_top5_acc])
            info_list = [epoch + 1, time_elapsed.total_seconds(), train_loss, train_top1_acc,
                         train_top5_acc]
            info_list = [str(i) for i in info_list]
            print("\t".join(info_list))
            # print(epoch + 1, time_elapsed.total_seconds(), train_loss, train_top1_acc,
            #       train_top5_acc)

            # tensorboardX
            # writer.add_scalar('learning rate', lr, epoch + 1)
            # writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
            # writer.add_scalars('accuracy', {'train accuracy': train_top1_acc, 'validation accuracy': val_top1_acc},
            #                    epoch + 1)

            is_best = train_top1_acc > best_prec1
            best_prec1 = max(train_top1_acc, best_prec1)

        # logger.close()
        # logger.plot()
        # savefig(os.path.join(args.checkpoint, 'log.eps'))
        # writer.close()

        print('Best accuracy:')
        print(best_prec1)



if __name__ == '__main__':
    main()
