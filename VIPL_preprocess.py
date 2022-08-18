# -*- coding:utf-8 -*-
# @File   : VIPL_preprocess.py
# @Time   : 2022/5/17 16:39
# @Author : Zhang Xinyu
import os
import warnings
import cv2
import numpy as np
from scipy import signal
import dlib
import pandas as pd
import tqdm
import argparse
from torch.utils.data import Dataset
from libs.py37_win import tttt
# from libs.py37_linux import tttt  # for linux.
from utils.cwtbag import cwt_filtering

warnings.filterwarnings("ignore")
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

def get_args_parser():
    parser = argparse.ArgumentParser('VIPL preprocessing', add_help=False)
    # Main params.
    parser.add_argument('--data-path', default='X:/vipl-hr', type=str,
                        help="""Please specify path to the 'vipl-hr' as input.""")
    parser.add_argument('--infos-path', default='./data/vipl-hr-infos', type=str,
                        help="""Please specify path to the 'vipl-hr-infos' as input.""")

    parser.add_argument('--frame-path', default='F:/vipl-frame/frame_list', type=str,
                        help="""Please specify path to the 'frame_list' as output.""")
    parser.add_argument('--mask-path', default='F:/vipl-frame/mask_list', type=str,
                        help="""Please specify path to the 'mask_list' as output.""")
    parser.add_argument('--wave-path', default='F:/vipl-frame/wave_gt', type=str,
                        help="""Please specify path to the 'wave' as output.""")
    parser.add_argument('--face-data-path', default='F:/vipl-face/data', type=str,
                        help="""Please specify path to the 'face_data' as output.""")
    parser.add_argument('--face-img-path', default='F:/vipl-face/img', type=str,
                        help="""Please specify path to the 'face_img' as output.""")
    return parser

def the_only_face(frame_in, scale=3, maxlenth=40):
    for i in range(len(frame_in)):
        rects = detector(frame_in[i], 0)
        lens = len(rects)
        if lens == 0:
            the_only_rect = (False, None)
        elif lens == 1:
            the_only_rect = (True, rects[0].rect)
        else:
            axis_x = int(frame_in[i].shape[0] / 2)
            axis_y = int(frame_in[i].shape[1] / 2)
            distances = [0.0 for x in range(lens)]
            for i in range(lens):
                rects_axis_x = int((rects[i].rect.right() - rects[i].rect.left()) / 2)
                rects_axis_y = int((rects[i].rect.right() - rects[i].rect.left()) / 2)
                distances[i] = (rects_axis_x - axis_x) ** 2 + (rects_axis_y - axis_y) ** 2

            min_distance_index = distances.index(min(distances))
            the_only_rect = (True, rects[min_distance_index].rect)
        if the_only_rect[0]:
            the_just_face_rect = the_only_rect[1]
            t0, b0, l0, r0 = the_just_face_rect.top(), the_just_face_rect.bottom(), the_just_face_rect.left(), the_just_face_rect.right()
            maxh = (b0 - t0) / scale
            maxw = (r0 - l0) / scale
            finaladdh = max(maxh, maxlenth)
            finaladdw = max(maxw, maxlenth)
            tf, bf, lf, rf = max(0, t0 - int(finaladdh / 2)), min(frame_in[i].shape[0], b0 + int(finaladdh / 2)), \
                             max(0, l0 - int(finaladdw / 2)), min(frame_in[i].shape[1], r0 + int(finaladdw / 2))
            return tf, bf, lf, rf
        else:
            continue
    return None


def filter(wave):
    f1 = 0.65
    f2 = 3.5
    samplingrate = 60
    b, a = signal.butter(6, [2 * f1 / samplingrate, 2 * f2 / samplingrate], 'bandpass')
    meanslist_after_BP = signal.filtfilt(b, a, np.array(wave))
    meanslist_after_BP = cwt_filtering(meanslist_after_BP, 60)[0]
    return meanslist_after_BP


class Dataset_vipl_hr_generate(Dataset):
    def __init__(self, args, person_number, task_number, frame_drop=12, prefix=0, random_shake=True,
                 cache_abandon=True):
        super().__init__()
        self.args = args
        self.buffer = 10
        self.image_size = 131
        self.margin = 20
        # self.mask_size = 64
        self.video_path = args.data_path
        self.task_number = task_number
        self.info_pool = []
        self.collect_info(args.infos_path, person_number)
        self.frame_drop = frame_drop
        self.video_check()
        self.wave_check()
        if cache_abandon:
            self.del_cache()
        self.random_shake = random_shake
        self.prefix = prefix

    def __getitem__(self, index):
        info = self.info_pool[index]
        txt_path = info[0]
        person_number = info[1]
        task_number = info[2]
        # only use "source2".
        video_path = os.path.join(self.video_path, os.path.join(os.path.join(person_number, task_number), "source2"))
        video_name = os.path.join(video_path, "video.avi")  # input
        wave_path = os.path.join(video_path, "wave.csv")  # label

        with open(txt_path, "r") as f:
            info_str = f.readline()
            start_place, end_place, hr = info_str.split("_")[0], info_str.split("_")[1], info_str.split("_")[2]
            start_place, end_place = int(start_place), int(end_place)
            start_place, end_place = start_place + self.buffer, end_place - self.buffer

            i = str(self.prefix)
            mask_list, residual_list, frame_list = self.read_video(start_place, end_place, video_name,
                                                                   person_number, task_number, i)
            wave_return_new = self.read_wave_csv(wave_path, start_place, end_place, start_place, end_place)

            save_path = os.path.join(self.args.frame_path, '_'.join([i, person_number, task_number, 's2',
                                                                           str(start_place),
                                                                           str(end_place), hr]) + '_.npy')
            frame_list_save = np.array(frame_list)
            np.save(save_path, frame_list_save)

            save_path = os.path.join(self.args.mask_path, '_'.join([i, person_number, task_number, 's2',
                                                                          str(start_place),
                                                                          str(end_place), hr]) + '_.npy')
            mask_list_save = np.array(mask_list)
            np.save(save_path, mask_list_save)

            save_path = os.path.join(self.args.wave_path, '_'.join([i, person_number, task_number, 's2',
                                                                        str(start_place),
                                                                        str(end_place), hr]) + '_.npy')
            wave_return_new_save = wave_return_new
            np.save(save_path, wave_return_new_save)
            self.prefix += 1
        return

    def __len__(self):
        return len(self.info_pool)

    def collect_info(self, info_path, person_number):
        for fn in os.listdir(info_path):
            if fn in person_number:
                total_dir = os.path.join(info_path, fn)
                for txt_name in os.listdir(total_dir):
                    version = txt_name.split('.')[0]
                    if version in self.task_number:
                        self.info_pool.append((os.path.join(total_dir, txt_name), fn, version))

    def video_check(self):
        del_list = []
        for index in range(len(self.info_pool)-1, -1, -1):
            info = self.info_pool[index]
            txt_path = info[0]
            person_number = info[1]
            task_number = info[2]
            video_path = os.path.join(self.video_path,
                                      os.path.join(os.path.join(person_number, task_number), "source2"))
            video_name = os.path.join(video_path, "video.avi")

            with open(txt_path, "r") as f:
                info_str = f.readline()
                start_place, end_place, _ = info_str.split("_")[0], info_str.split("_")[1], info_str.split("_")[2]
                start_place, end_place = int(start_place), int(end_place)
                start_place, end_place = start_place + self.buffer, end_place - self.buffer

            cap = cv2.VideoCapture(video_name)
            total_ticks = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_ticks <= end_place:
                del_list.append(index)

        for index in del_list:
            print('video invalid, del: {}'.format(self.info_pool[index]))
            self.info_pool.pop(index)
    def wave_check(self):
        del_list = []
        for index in range(len(self.info_pool)-1, -1, -1):
            info = self.info_pool[index]
            txt_path = info[0]
            person_number = info[1]
            task_number = info[2]
            video_path = os.path.join(self.video_path,
                                      os.path.join(os.path.join(person_number, task_number), "source2"))
            wave_path = os.path.join(video_path, "wave.csv")

            with open(txt_path, "r") as f:
                info_str = f.readline()
                start_place, end_place, _ = info_str.split("_")[0], info_str.split("_")[1], info_str.split("_")[2]
                start_place, end_place = int(start_place), int(end_place)
                start_place, end_place = start_place + self.buffer, end_place - self.buffer

            wave_csv_name = os.path.join(wave_path)
            wave_bvp = pd.read_csv(wave_csv_name)["Wave"].values
            start_place, end_place = start_place * 2, end_place * 2
            wave_return = wave_bvp[start_place: end_place]

            wave_return = filter(wave_return)
            wave_return_new = []

            for i in range(len(wave_return)):
                if i % 2 == 0:
                    wave_return_new.append(wave_return[i])

            if len(wave_return) < end_place - start_place:
                del_list.append(index)

        del_list = list(del_list)
        sorted(del_list, reverse=True)

        for index in del_list:
            print('wave invalid, del: {}'.format(self.info_pool[index]))
            self.info_pool.pop(index)


    def read_wave_csv(self, wave_path, start_place_this_time, end_place_this_time, start_place, end_place):
        wave_csv_name = os.path.join(wave_path)
        wave_bvp = pd.read_csv(wave_csv_name)["Wave"].values

        start_place, end_place = start_place*2, end_place*2
        wave_return = wave_bvp[start_place: end_place]
        wave_return = filter(wave_return)
        wave_return_new = []
        for i in range(len(wave_return)):
            if i % 2 == 0:
                wave_return_new.append(wave_return[i])

        wave_return_new = wave_return_new[start_place_this_time-int(start_place/2): end_place_this_time-int(start_place/2)]
        wave_return_output = wave_return_new[:]
        return wave_return_output

    def read_video(self, start_place_this_time, end_place_this_time, video_name, person_number, task_number, i):
        time_slice_this_time = range(start_place_this_time, end_place_this_time)
        cap = cv2.VideoCapture(video_name)
        success = True
        frame_list = []
        mask_list = []
        tick_count = 0
        while success:
            success, frame = cap.read()
            if frame is not None:
                tick_count += 1
                if tick_count in time_slice_this_time:
                    frame_list.append(frame)
            if len(frame_list) >= len(time_slice_this_time):
                break
        tf, bf, lf, rf = the_only_face(frame_list)  # top, bottom, left, right

        if self.random_shake:
            shape_of_frame = frame_list[0].shape
            tf, bf, lf, rf = self.random_shake_frame(tf, bf, lf, rf, shape_of_frame)

        save_path = os.path.join(self.args.face_data_path, '_'.join([i, person_number, task_number, 's2',
                                                                str(start_place_this_time),
                                                                str(end_place_this_time)]) + '.npy')
        np.save(save_path, [tf, bf, lf, rf])

        img = frame_list[0].copy()
        cv2.rectangle(img, (lf, tf), (rf, bf), (0, 0, 255), 3)
        save_path = os.path.join(self.args.face_img_path, '_'.join([i, person_number, task_number, 's2',
                                                               str(start_place_this_time),
                                                               str(end_place_this_time)]) + '.png')
        cv2.imwrite(save_path, img)

        for i in range(len(frame_list)):
            frame_list[i] = cv2.resize(frame_list[i][tf: bf, lf: rf], (self.image_size + self.margin, self.image_size + self.margin))

        for i in range(len(frame_list)):
            mask_list.append((tttt.RGBskin(cv2.resize(frame_list[i], (self.image_size + self.margin, self.image_size + self.margin))) / 255.0).astype(np.uint8))

        residual_list = []
        mask_list = mask_list[:-1]

        for i in range(len(frame_list) - 1):
            image = frame_list[i + 1].astype(np.int32) - frame_list[i].astype(np.int32)
            residual_list.append(image)

        return mask_list, residual_list, frame_list

    def random_shake_frame(self, tf, bf, lf, rf, shape_of_frame):
        # change_size = np.random.randint(0, 5)
        # choice = np.random.random()
        # if 0 <= choice < 0.3:
        #     tf, bf, lf, rf = tf - change_size, bf + change_size, lf - change_size, rf + change_size
        # elif 0.3 <= choice < 0.6:
        #     tf, bf, lf, rf = tf + change_size, bf - change_size, lf + change_size, rf - change_size
        # elif 0.6 <= choice < 0.7:
        #     tf, bf, lf, rf = tf + change_size, bf + change_size, lf, rf
        # elif 0.7 <= choice < 0.8:
        #     tf, bf, lf, rf = tf - change_size, bf - change_size, lf, rf
        # elif 0.8 <= choice < 0.9:
        #     tf, bf, lf, rf = tf, bf, lf + change_size, rf + change_size
        # else:
        #     tf, bf, lf, rf = tf, bf, lf - change_size, rf - change_size

        tf, bf, lf, rf = tf - self.margin, bf + self.margin, lf - self.margin, rf + self.margin

        tf = max(0, tf)
        lf = max(0, lf)
        rm = shape_of_frame[1]
        bm = shape_of_frame[0]
        bf = min(bf, bm)
        rf = min(rf, rm)
        return tf, bf, lf, rf

    def del_cache(self):
        paths = [self.args.face_data_path, self.args.face_img_path, self.args.frame_path, self.args.mask_path, self.args.wave_path]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
            ls = os.listdir(path)
            for i in ls:
                c_path = os.path.join(path, i)
                os.remove(c_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VIPL preprcessing', parents=[get_args_parser()])
    args = parser.parse_args()
    total_set = set(list(range(108)))
    version_type = [rf"v{i}" for i in range(1, 10)]
    person_name = [rf"p{i}" for i in total_set]
    datasets = Dataset_vipl_hr_generate(args, person_name, version_type, prefix=0, random_shake=True, cache_abandon=True)
    for i in tqdm.tqdm(range(len(datasets))):
        datasets.__getitem__(i)


