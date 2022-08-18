import torch
from torch import nn

# 用于计算2维的fft
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


# 用于计算1维的fft
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


# 这个函数的输入是一维的，不带batch维度的等待fft的序列。
class Reg_version_1(nn.Module):
    #这个正则化的方法的核心是，理论上每一点和另外一点的关系，只和距离有关。所以他们之间的值应当很接近。
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
            #计算每个距离集合的std,通常来说，std越小，代表这一层级越发稳定。这个len(distance_record[i])表示如果
            #聚类里面的成员越多，这个std的权重就越大。除以5是防止这个数太大导致loss很大。
        # for i in range(len(result)):
        #     print(len(result[i]))
        # print(result)
        amassed = torch.mean(torch.cat(result, 0))
        return amassed


# 第二个正则化，每一斜行的均值组成的波形应该尽量逼近正弦函数。
# 所以这个类的目的是将一张图根据斜行的均值化为一个类似于正弦函数的波形。
class Turn_map_into_waves(nn.Module):
    # 第二个正则化，每一斜行的均值组成的波形应该尽量逼近正弦函数。
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
            result.append(torch.mean(torch.cat(distance_record[i], 0)).unsqueeze(0)) #计算均值！let's see!
        amassed = torch.cat(result, 0)
        return amassed


# 对于生成的波形的正则化（输入是attn的batch，输出一个wave的标准）
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
        #选择一个最大的坐标。
        max_index = torch.argmax(wave_altered[select_space[0]:select_space[1]], 0) + select_space[0]
        judgement = 1 - (wave_altered[max_index] / torch.sum(wave_altered[select_space[0]:select_space[1]]))
        return judgement


if __name__ == "__main__":
    pass