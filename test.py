# -*- coding:utf-8 -*-
# @File   : test.py
# @Time   : 2022/7/25 15:33
# @Author : Zhang Xinyu
import os
import time
import warnings
import argparse
import logging

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from models.model import Ultimate_model
from torch.utils.data import DataLoader
from losses.loss import ATTN_LOSS_Pearson, Mask_loss
from datasets import Dataset_VIPL_HR_Offline
from utils.hr_calc import hr_cal, peakcheckez
from utils.fft_package import Reg_version_wave, Turn_map_into_waves

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('BYHE testing', add_help=False)
    # Main params.
    parser.add_argument('--frame-path', default='./data/vipl-frame/frame_list', type=str,
                        help="""Please specify path to the 'frame_list' as input.""")
    parser.add_argument('--mask-path', default='./data/vipl-frame/mask_list', type=str,
                        help="""Please specify path to the 'mask_list' as GT.""")
    parser.add_argument('--wave-path', default='./data/vipl-frame/wave_gt', type=str,
                        help="""Please specify path to the 'wave' as GT.""")
    parser.add_argument('--length', default=70, type=int, help="""Length for video frames training.""")
    parser.add_argument('--test-length', default=300, type=int, help="""Length for video frames testing (HR Calculate).""")
    parser.add_argument('--win-length', default=11, type=int, help="""Sliding window length. (default: 11)""")
    parser.add_argument('--GPU-id', default=0, type=int, help="""Index of GPUs.""")
    parser.add_argument('--num-workers', default=0, type=int, help="""Number of data loading workers per GPU. (default: 
                        0)""")
    parser.add_argument('--log-enable', default=True, type=bool, help="""Whether or not enable tensorboard and logging. 
                       (Default: True).""")
    parser.add_argument('--visual-enable', default=False, type=bool, help="""Whether or not enable plt visualization. 
                       (Default: True).""")
    parser.add_argument('--pretrained', type=str, default='./pretrained/VIPL_f1.pth', help='pretrained weights path.')
    return parser


def test(args):
    # ============ Setup logging ... ============
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Start testing at', start_time, end='\n\n')

    if args.log_enable:
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)

        os.makedirs('./logs', exist_ok=True)
        logging.basicConfig(filename=f'./logs/BYHE_test_GPU{args.GPU_id}.log', filemode='w', level=logging.INFO,
                            format='%(levelname)s: %(message)s')
        logging.info('Start testing at {}\n'.format(start_time))

    # ============ preparing data ... ============
    test_set = set(list(range(0, 22)))
    version_type = [rf"v{i}" for i in range(1, 10)]
    person_name = [rf"p{i}" for i in test_set]
    test_set = Dataset_VIPL_HR_Offline(args.frame_path, person_name, version_type, args.mask_path, args.wave_path, length=args.test_length,
                                       is_train=False)

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,  # set 'pin_memory=True' if you have enough RAM.
        shuffle=False
    )

    print(f"Data loaded: there are {len(test_set)} videos for testing.")
    if args.log_enable:
        logging.info(f"Data loaded: there are {len(test_set)} videos for testing.")

    # ============ building model ... ============
    model = Ultimate_model(args).to(torch.device(args.GPU_id))
    dict = torch.load(args.pretrained, map_location='cpu')
    msg = model.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model'], strict=False)
    print('Pretrained weights found at {}'.format(args.pretrained))
    print('load_state_dict msg: {}'.format(msg))
    print(f"saved epoch: {dict['epoch']}", end='\n\n')
    if args.log_enable:
        logging.info('Pretrained weights found at {}'.format(args.pretrained))
        logging.info('load_state_dict msg: {}'.format(msg))
        logging.info(f"saved epoch: {dict['epoch']}\n")

    model.eval()

    # ============ preparing loss ... ============
    Loss_atten = ATTN_LOSS_Pearson()
    Loss_mask = Mask_loss()
    Loss_MSE = nn.MSELoss()
    Reg_wave = Reg_version_wave(args.GPU_id)  # Regularization

    # ============ testing ... ============
    with torch.no_grad():
        hr_MAE_total = []
        cnt = 0
        for it, (input_residual, input_frame, test_attn_label, skin_mask, wave_label, _, _, _, name) in enumerate(test_loader):
            cnt += 1
            input_residual = input_residual.to(torch.device(args.GPU_id))
            input_frame = input_frame.to(torch.device(args.GPU_id))
            attn_raw, output_mask, output_mask2 = model(input_residual, input_frame)

            # loss calculate.
            wave_of_attn_raw = Turn_map_into_waves()(attn_raw)
            wave_of_attn_label = Turn_map_into_waves()(test_attn_label.to(torch.device(args.GPU_id)))
            loss_atten = Loss_atten(attn_raw, test_attn_label.to(torch.device(args.GPU_id)))
            loss_mse = Loss_MSE(attn_raw, test_attn_label.to(torch.device(args.GPU_id)))
            loss_mask = Loss_mask(output_mask, skin_mask.to(torch.device(args.GPU_id)))
            loss_reg = Reg_wave(attn_raw)

            total_loss = (
                0.8 * loss_atten +
                1.0 * loss_mse +
                0.2 * loss_mask +
                0.1 * loss_reg
            )

            hr_train, altered_wave = hr_cal(wave_of_attn_raw.detach().cpu().numpy().tolist()[0])
            wave_label_per_sample = wave_label[0]
            hr_label = peakcheckez(wave_label_per_sample, 30)
            hr_MAE = abs(hr_train - hr_label)
            hr_MAE_total.append(hr_MAE)

            print(f'({cnt}/{len(test_set)})\t[{name[0]}]   \t Total Loss: {total_loss.item():.4f}\t||\tHR train: {hr_train:.4f}\t||\tHR label: {hr_label:.4f}\t||\tMAE bpm: {hr_MAE:.4f} ({np.mean(np.array(hr_MAE_total)):.4f})')
            if args.log_enable:
                logging.info(f'({cnt}/{len(test_set)})\t[{name[0]}]   \t Total Loss: {total_loss.item():.4f}\t||\tHR train: {hr_train:.4f}\t||\tHR label: {hr_label:.4f}\t||\tMAE bpm: {hr_MAE:.4f} ({np.mean(np.array(hr_MAE_total)):.4f})')

            if args.visual_enable:
                plt.figure(figsize=(12,6))
                plt.subplot(2, 2, 1)
                plt.imshow(attn_raw[0].clone().detach().cpu().numpy())
                plt.colorbar()

                plt.subplot(2, 2, 2)
                plt.plot(wave_of_attn_raw[0].clone().detach().cpu().numpy(), label='output_self_similarity_wave')
                plt.plot(wave_of_attn_label[0].clone().detach().cpu().numpy(), label='gt_self_similarity_wave')
                plt.legend()

                plt.subplot(2, 2, 3)
                altered_wave /= np.max(np.abs(altered_wave))
                wave_label_per_sample /= np.max(np.abs(wave_label_per_sample))
                plt.plot(altered_wave, label='output_self_similarity_wave_filtered&normalized')
                plt.plot(wave_label_per_sample, label='gt_self_similarity_wave_filtered&normalized')
                plt.legend()

                plt.subplot(2, 2, 4)
                plt.imshow(output_mask[0][0].clone().detach().cpu().numpy())
                plt.colorbar()

                plt.suptitle(f'[{name[0]}]  Total loss: {total_loss:.4f}, MAE: {hr_MAE:.2f} bpm', fontsize=15)
                plt.show()

    finish_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Finish testing at', finish_time)
    if args.log_enable:
        logging.info('Finish testing at {}\n'.format(finish_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BYHE testing', parents=[get_args_parser()])
    args = parser.parse_args()
    test(args)