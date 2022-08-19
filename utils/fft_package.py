# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
import torch
from torch import nn

class FFT_MODULE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        batch_list = torch.split(input, 1, 0)
        result_list = []
        for i in range(len(batch_list)):
            attn_zeros = torch.zeros(batch_list[i].shape[1:]).unsqueeze(-1).cuda()
            attn_real_and_image = torch.cat([batch_list[i].squeeze().unsqueeze(-1), attn_zeros], -1)
            result = torch.fft(attn_real_and_image, 2)
            final_result = torch.norm(result, p=2, dim=2).unsqueeze(0)
            result_list.append(final_result)

        result = torch.cat(result_list, 0)
        return result


class FFT_MODULE_1d(nn.Module):
    def __init__(self, GPU_id, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.GPU_id = GPU_id

    def forward(self, input):
        pass

    def solo_fft_1d_make(self, input):
        if self.use_cuda:
            attn_zeros = torch.zeros(input.shape).unsqueeze(-1).to(torch.device(self.GPU_id))
        else:
            attn_zeros = torch.zeros(input.shape).unsqueeze(-1)
        attn_real_and_image = torch.cat([input.squeeze().unsqueeze(-1), attn_zeros], -1)
        result = torch.fft(attn_real_and_image, 1)
        final_result = torch.norm(result, p=2, dim=1)
        return final_result


class Reg_version_1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn):
        result_list = []
        for i in range(attn.shape[0]):
            result_list.append(self.attn_solo_process(attn[i]).unsqueeze(0))
        result = torch.mean(torch.cat(result_list, 0))
        return result

    def attn_solo_process(self, attn_solo):
        distance_record = [[] for x in range(attn_solo.shape[0]-1)]
        # print(len(distance_record))
        position_check = [[] for x in range(attn_solo.shape[0]-1)]
        for i in range(attn_solo.shape[0] - 1):
            for j in range(i+1, attn_solo.shape[1]):
                # print(j-i-1)
                position_check[j - i - 1].append([i, j])
                distance_record[j-i-1].append(attn_solo[i, j].unsqueeze(0))
        # print(position_check)
        # for i in range(len(distance_record)):
        #     print(len(distance_record[i]))
        # # print(distance_record)
        result = []
        for i in range(len(distance_record) - 1):
            # result.append(torch.cat(distance_record[i], 0))
            # print(len(distance_record[i]))
            result.append(torch.std(torch.cat(distance_record[i], 0) * len(distance_record[i])/5).unsqueeze(0))
        # for i in range(len(result)):
        #     print(len(result[i]))
        # print(result)
        amassed = torch.mean(torch.cat(result, 0))
        return amassed

class Turn_map_into_waves(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn):
        result_list = []
        for i in range(attn.shape[0]):
            result_list.append(self.attn_solo_process(attn[i]).unsqueeze(0))
        result = torch.cat(result_list, 0)
        return result

    def attn_solo_process(self, attn_solo):
        distance_record = [[] for x in range(attn_solo.shape[0])]
        position_check = [[] for x in range(attn_solo.shape[0])]
        for i in range(attn_solo.shape[0]):
            for j in range(i, attn_solo.shape[1]):
                position_check[j - i].append([i, j])
                distance_record[j - i].append(attn_solo[i, j].unsqueeze(0))

        result = []
        for i in range(len(distance_record)):
            result.append(torch.mean(torch.cat(distance_record[i], 0)).unsqueeze(0))
        amassed = torch.cat(result, 0)
        return amassed


class Reg_version_wave(nn.Module):
    def __init__(self, Gpu_ID):
        super().__init__()
        self.fft_module_turner = FFT_MODULE_1d(Gpu_ID)
        self.map_to_wave = Turn_map_into_waves()

    def forward(self, attns):
        waves = self.map_to_wave(attns)
        wave_list = torch.split(waves, 1, 0)
        result = []
        for i in range(len(wave_list)):
            # print(wave_list[i].shape)
            tmp = self.solo_reg_make(wave_list[i].squeeze(0)).unsqueeze(0)
            result.append(tmp)

        final_result = torch.mean(torch.cat(result, 0))
        return final_result

    def solo_reg_make(self, input):
        wave_altered = self.fft_module_turner.solo_fft_1d_make(input)
        lens = wave_altered.shape[0]
        select_space = (1, 1 + int(lens/2))
        max_index = torch.argmax(wave_altered[select_space[0]:select_space[1]], 0) + select_space[0]
        judgement = 1 - (wave_altered[max_index] / torch.sum(wave_altered[select_space[0]:select_space[1]]))
        return judgement


if __name__ == "__main__":
    pass