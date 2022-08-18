# -*- coding:utf-8 -*-
# @File   : train.py
# @Time   : 2022/8/4 10:06
# @Author : Zhang Xinyu
import os
import time
import sys
import warnings
import argparse
import math
from pathlib import Path
import logging

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from datasets import Dataset_VIPL_HR_Offline

from models.model import Ultimate_model
from losses.loss import ATTN_LOSS_Pearson, Mask_loss
from utils.hr_calc import hr_cal, peakcheckez
from utils.fft_package import Reg_version_wave, Turn_map_into_waves
from utils.utils import cosine_scheduler, LARS, restart_from_checkpoint
from utils.metric_logger import AverageMeter, ProgressMeter
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('BYHE training', add_help=False)
    # Main params.
    parser.add_argument('--frame-path', default='X:/vipl-frame/frame_list', type=str,
                        help="""Please specify path to the 'frame_list' as input.""")
    parser.add_argument('--mask-path', default='X:/vipl-frame/mask_list', type=str,
                        help="""Please specify path to the 'mask_list' as GT.""")
    parser.add_argument('--wave-path', default='X:/vipl-frame/wave_gt', type=str,
                        help="""Please specify path to the 'wave' as GT.""")
    parser.add_argument('--length', default=70, type=int, help="""Length of video frames.""")
    parser.add_argument('--test-length', default=300, type=int, help="""Length for video frames testing (HR Calculate).""")
    parser.add_argument('--win-length', default=11, type=int, help="""Sliding window length. (default: 11)""")
    parser.add_argument('--output-dir', default='./saved/', type=str, help="""Path to save logs and checkpoints.""")
    parser.add_argument('--GPU-id', default=0, type=int, help="""Index of GPUs.""")
    parser.add_argument('--saveckp-freq', default=10, type=int, help="""Save checkpoint every x epochs.""")
    parser.add_argument('--num-workers', default=0, type=int, help="""Number of data loading workers per GPU. (default: 
                        0)""")
    parser.add_argument('--batch-size', default=1, type=int, help="""batch-size: number of distinct images loaded on GPU.
                        (default: 6)""")
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training. (default: 50)')
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of linear warmup (highest LR 
                        used during training). The learning rate is linearly scaled with the batch size, and specified here
                        for a reference batch size of 256. (default: 1e-3)""")
    parser.add_argument('--min-lr', default=1e-6, type=float, help="""Target LR at the end of optimization. We use a cosine
                        LR schedule with linear warmup. (default: 1e-5)""")
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adamw', 'adam', 'sgd', 'lars'], help="""Type of 
                        optimizer. We recommend using adamw with ViTs. (default: 'adamw')""")
    parser.add_argument("--warmup-epochs", default=0, type=int, help="""Number of epochs for the linear learning-rate warm
                         up. (default: 20)""")
    parser.add_argument('--weight-decay', default=1e-5, type=float, help="""Initial value of the weight decay. With ViT, a 
                        smaller value at the beginning of training works well. (default: 1e-5)""")
    parser.add_argument('--weight-decay-end', default=1e-3, type=float, help="""Final value of the weight decay. We use a 
                        cosine schedule for WD and using a larger decay by the end of training improves performance for 
                        ViTs. (default: 1e-3)""")
    parser.add_argument('--log-enable', default='True', type=str, help="""Whether or not enable tensorboard and logging. 
                       (Default: True).""")
    parser.add_argument('--print-freq', default=1, type=int, help="""Print metrics every x iterations.""")
    parser.add_argument('--log-theme', default='VIPL', type=str, help="""Annotation for tensorboard.""")
    return parser


def train(args):
    cudnn.benchmark = True
    # ============ Setup logging ... ============
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Start training at', start_time, end='\n\n')
    if args.log_enable=='True':
        tb_writer = SummaryWriter(f'./tensorboard/BYHE_GPU{args.GPU_id}',
                                  filename_suffix=f'_BYHE_GPU{args.GPU_id}')
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        logging.basicConfig(filename=f'./logs/BYHE_GPU{args.GPU_id}.log', filemode='w', level=logging.INFO,
                            format='%(levelname)s: %(message)s')
        logging.info('Start training at {}\n'.format(start_time))

        # logging hyperparameters
        logging.info("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args)).items()) + '\n')
    else:
        tb_writer = None
    print("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args)).items()), end='\n\n')

    # ============ preparing data ... ============
    total_set = set(list(range(108)))
    test_set = set(list(range(0, 22)))

    train_set = total_set - test_set
    version_type = [rf"v{i}" for i in range(1, 10)]
    person_name = [rf"p{i}" for i in train_set]
    train_set = Dataset_VIPL_HR_Offline(args.frame_path, person_name, version_type, args.mask_path, args.wave_path, length=args.length,
                                        is_train=True)

    version_type = [rf"v{i}" for i in range(1, 10)]
    person_name = [rf"p{i}" for i in test_set]
    test_set = Dataset_VIPL_HR_Offline(args.frame_path, person_name, version_type, args.mask_path, args.wave_path, length=args.test_length,
                                       is_train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,  # set 'pin_memory=True' if you have enough RAM.
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,  # set 'pin_memory=True' if you have enough RAM.
    )
    print(f"Data loaded: there are {len(train_set)} videos for training, {len(test_set)} videos for testing.")

    # ============ building model ... ============
    model = Ultimate_model(args).to(torch.device(args.GPU_id))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: {:.2f}M".format(total / 1e6), end='\n\n')

    # ============ preparing loss ... ============
    loss_atten = ATTN_LOSS_Pearson()
    loss_mask = Mask_loss()
    loss_MSE = nn.MSELoss()
    loss_reg = Reg_version_wave(args.GPU_id)  # Regularization
    loss_mapping = {
        'loss_atten': 0,
        'loss_mask': 1,
        'loss_MSE': 2,
        'loss_reg': 3,
    }
    losses = (loss_atten, loss_mask, loss_MSE, loss_reg, loss_mapping)

    # ============ preparing optimizer ... ============
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parameters)  # to use with ViTs
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr = 0)  # lr is set by scheduler
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr = 0, momentum = 0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = LARS(parameters)  # to use with convnet and large batches

    # ============ init schedulers ... ============
    lr_schedule = cosine_scheduler(
        # args.lr * (args.batch_size / 256.),  # linear scaling rule
        args.lr,
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.output_dir, f'BYHE_best_GPU{args.GPU_id}.pth'),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
    )

    start_epoch = to_restore["epoch"]

    print("Starting training !")
    best_mae = float('inf')
    for epoch in range(start_epoch, args.epochs):
        # ============ training one epoch ... ============
        train_one_epoch(model, losses, train_loader, optimizer, lr_schedule, wd_schedule, epoch, logging, tb_writer, args)
        # ============ validating ... ============
        avg_mae = validate(model, losses, test_loader, epoch, logging, tb_writer, args)
        # ============ save weights ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args
        }
        if epoch % args.saveckp_freq == 0 or epoch == args.epochs-1:
            torch.save(save_dict, os.path.join(args.output_dir, f'BYHE_checkpoint{epoch:04}_GPU{args.GPU_id}.pth'))
        # remember and save the best checkpoint.
        if avg_mae < best_mae:
            best_mae = min(avg_mae, best_mae)
            torch.save(save_dict, os.path.join(args.output_dir, f'BYHE_best_GPU{args.GPU_id}.pth'))

    finish_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Finish training at', finish_time)


def train_one_epoch(model, criterions, data_loader, optimizer, lr_schedule, wd_schedule, epoch, logging, tb_writer, args):
    # metric init.
    batch_time_metric = AverageMeter('Time', ':6.3f')
    total_loss_metric = AverageMeter('Total Loss', ':5.3f')
    loss_atten_metric = AverageMeter(' Loss Atten', ':5.3f')
    loss_mse_metric = AverageMeter('  Loss MSE', ':5.3f')
    loss_mask_metric = AverageMeter(' Loss Mask', ':5.3f')
    loss_reg_metric = AverageMeter(' Loss Reg', ':5.3f')
    lr_metric = AverageMeter('   lr', ':6.4f')
    wd_metric = AverageMeter(' wd', ':6.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time_metric,
         total_loss_metric,
         loss_atten_metric,
         loss_mse_metric,
         loss_mask_metric,
         loss_reg_metric,
         lr_metric,
         wd_metric],
        prefix="Epoch: [{}]".format(epoch),
        log_enable=args.log_enable
    )

    model.train()

    end = time.time()
    for it, (inputs, real_image_train, train_attn_label, skin_mask, wave_label, path, start_f, end_f, _) in enumerate(data_loader):
        # update weight decay and learning rate according to their schedule.
        train_iter = len(data_loader) * epoch + it  # global training iteration.
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[train_iter]
            if i == 0:  # only the first group is regularized.
                param_group["weight_decay"] = wd_schedule[train_iter]

        inputs = inputs.to(torch.device(args.GPU_id))
        real_image_train = real_image_train.to(torch.device(args.GPU_id))

        attn_raw, output_mask2, _ = model(inputs, real_image_train)

        if args.log_enable=='True' and (epoch % 10 == 0 or epoch == args.epochs - 1):
            for b in range(attn_raw.shape[0]):  # batch size
                tb_writer.add_image(
                    '_'.join(['train-output', path[b].split('_')[0], path[b].split('_')[1], path[b].split('_')[2],
                              path[b].split('_')[3], str(start_f[b].item()), str(end_f[b].item())]),
                    (attn_raw[b] * 127.5 + 127.5).unsqueeze(dim=0).detach().cpu().numpy().astype(np.uint8), epoch)
                tb_writer.add_image(
                    '_'.join(['train-label', path[b].split('_')[0], path[b].split('_')[1], path[b].split('_')[2],
                              path[b].split('_')[3], str(start_f[b].item()), str(end_f[b].item())]),
                    (train_attn_label[b] * 127.5 + 127.5).unsqueeze(dim=0).detach().cpu().numpy().astype(np.uint8), epoch)

        # loss calculate.
        loss_mapping = criterions[-1]
        loss_atten = criterions[loss_mapping['loss_atten']](attn_raw, train_attn_label.to(torch.device(args.GPU_id)))
        loss_mse = criterions[loss_mapping['loss_MSE']](attn_raw, train_attn_label.to(torch.device(args.GPU_id)))
        loss_mask = criterions[loss_mapping['loss_mask']](output_mask2, skin_mask.to(torch.device(args.GPU_id)))
        loss_reg = criterions[loss_mapping['loss_reg']](attn_raw)

        total_loss = (
            0.8 * loss_atten +
            1.0 * loss_mse +
            0.2 * loss_mask +
            0.1 * loss_reg
        )
        # ================= end of the amp calc. =================

        # metric update.
        total_loss_metric.update(total_loss.item(), attn_raw.shape[0])
        loss_atten_metric.update(loss_atten.item(), attn_raw.shape[0])
        loss_mse_metric.update(loss_mse.item(), attn_raw.shape[0])
        loss_mask_metric.update(loss_mask.item(), attn_raw.shape[0])
        loss_reg_metric.update(loss_reg.item(), attn_raw.shape[0])
        lr_metric.update(optimizer.param_groups[0]["lr"], attn_raw.shape[0])
        wd_metric.update(optimizer.param_groups[0]["weight_decay"], attn_raw.shape[0])

        if not math.isfinite(total_loss.item()):
            print("Total loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # estimate elapsed time.
        batch_time_metric.update(time.time() - end)
        end = time.time()

        if it % args.print_freq == 0:
            progress.display(it)

    print('Train Avg: Total Loss: {total_loss.avg:.4f}'.format(total_loss=total_loss_metric))

    # ============ writing logs ... ============
    if args.log_enable=='True':
        logging.info('Train Avg: Total Loss: {total_loss.avg:.4f}'.format(total_loss=total_loss_metric))
        tb_writer.add_scalars('train_loss', {'total_loss': total_loss_metric.avg,
                                             'map_pearson_loss': loss_atten_metric.avg,
                                             'map_mse_loss': loss_mse_metric.avg,
                                             'mask_mse_loss': loss_mask_metric.avg,
                                             'reg_loss': loss_reg_metric.avg,
                                             'learning_rate': lr_metric.avg,
                                             'weight_decay': wd_metric.avg}, epoch)


def validate(model, criterions, data_loader, epoch, logging, tb_writer, args):
    # metric init.
    batch_time_metric = AverageMeter('Time', ':6.3f')
    total_loss_metric = AverageMeter('Total Loss', ':5.3f')
    MAE_bpm_metric = AverageMeter('MAE bpm', ':6.2f')
    loss_atten_metric = AverageMeter('   Loss Atten', ':5.3f')
    loss_mse_metric = AverageMeter('   Loss MSE', ':5.3f')
    loss_mask_metric = AverageMeter(' Loss Mask', ':5.3f')
    loss_reg_metric = AverageMeter(' Loss Reg', ':5.3f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time_metric,
         total_loss_metric,
         MAE_bpm_metric,
         loss_atten_metric,
         loss_mse_metric,
         loss_mask_metric,
         loss_reg_metric],
        prefix="Test: ",
        log_enable=args.log_enable
    )

    model.eval()

    end = time.time()
    with torch.no_grad():
        hr_train_bpm = []
        hr_label_bpm = []
        for it, (inputs, real_image_test, test_attn_label, skin_mask, wave_label, path, start_f, end_f) in enumerate(data_loader):
            inputs = inputs.to(torch.device(args.GPU_id))
            real_image_test = real_image_test.to(torch.device(args.GPU_id))

            attn_raw, output_mask2, _ = model(inputs, real_image_test)

            if args.log_enable=='True' and (epoch % 10 == 0 or epoch == args.epochs - 1):
                for b in range(attn_raw.shape[0]):  # batch size
                    tb_writer.add_image('_'.join(['val-output', path[b].split('_')[0], path[b].split('_')[1], path[b].split('_')[2],
                                                  str(start_f[b].item()), str(end_f[b].item())]),
                                        (attn_raw[b] * 127.5 + 127.5).unsqueeze(dim=0).detach().cpu().numpy().astype(np.uint8), epoch)
                    tb_writer.add_image('_'.join(['val-label', path[b].split('_')[0], path[b].split('_')[1], path[b].split('_')[2],
                                                  str(start_f[b].item()), str(end_f[b].item())]),
                                        (test_attn_label[b] * 127.5 + 127.5).unsqueeze(dim=0).detach().cpu().numpy().astype(np.uint8), epoch)

            # loss calculate.
            wave_of_attn_raw = Turn_map_into_waves()(attn_raw)
            loss_mapping = criterions[-1]
            loss_atten = criterions[loss_mapping['loss_atten']](attn_raw, test_attn_label.to(torch.device(args.GPU_id)))
            loss_mse = criterions[loss_mapping['loss_MSE']](attn_raw, test_attn_label.to(torch.device(args.GPU_id)))
            loss_mask = criterions[loss_mapping['loss_mask']](output_mask2, skin_mask.to(torch.device(args.GPU_id)))
            loss_reg = criterions[loss_mapping['loss_reg']](attn_raw)

            total_loss = (
                0.8 * loss_atten +
                1.0 * loss_mse +
                0.2 * loss_mask +
                0.1 * loss_reg
            )

            mae_bpm_batch = []
            for i in range(attn_raw.shape[0]):
                hr_train, altered_wave = hr_cal(wave_of_attn_raw.detach().cpu().numpy().tolist()[i])
                wave_label_per_sample = wave_label[i]
                hr_label = peakcheckez(wave_label_per_sample, 30)
                hr_train_bpm.append(hr_train)
                hr_label_bpm.append(hr_label)
                mae_bpm_batch.append(abs(hr_label - hr_train))

            # metric update.
            total_loss_metric.update(total_loss.item(), attn_raw.shape[0])
            MAE_bpm_metric.update(np.mean(mae_bpm_batch), attn_raw.shape[0])
            loss_atten_metric.update(loss_atten.item(), attn_raw.shape[0])
            loss_mse_metric.update(loss_mse.item(), attn_raw.shape[0])
            loss_mask_metric.update(loss_mask.item(), attn_raw.shape[0])
            loss_reg_metric.update(loss_reg.item(), attn_raw.shape[0])

            # estimate elapsed time.
            batch_time_metric.update(time.time() - end)
            end = time.time()

            if it % args.print_freq == 0:
                progress.display(it)

    RMSE_bpm = mean_squared_error(hr_train_bpm, hr_label_bpm, squared=False)
    print('Test Avg: Total Loss: {total_loss.avg:.4f}  MAE bpm: {MAE_bpm.avg:.4f}  RMSE bpm: {RMSE_bpm:.4f}'.format(
              total_loss=total_loss_metric,
              MAE_bpm=MAE_bpm_metric,
              RMSE_bpm=RMSE_bpm), end='\n\n')
    if args.log_enable=='True':
        logging.info('Test Avg: Total Loss: {total_loss.avg:.4f}  MAE bpm: {MAE_bpm.avg:.4f}  RMSE bpm: {RMSE_bpm:.4f}\n'.format(
            total_loss=total_loss_metric,
            MAE_bpm=MAE_bpm_metric,
            RMSE_bpm=RMSE_bpm)
        )
        tb_writer.add_scalar('val_MAE_bpm_' + args.log_theme, MAE_bpm_metric.avg, epoch)
        tb_writer.add_scalar('val_RMSE_bpm_' + args.log_theme, RMSE_bpm, epoch)
        tb_writer.add_scalars('val_loss', {'total_loss': total_loss_metric.avg,
                                           'map_pearson_loss': loss_atten_metric.avg,
                                           'map_mse_loss': loss_mse_metric.avg,
                                           'mask_mse_loss': loss_mask_metric.avg,
                                           'reg_loss': loss_reg_metric.avg}, epoch)
    return MAE_bpm_metric.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BYHE training', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)