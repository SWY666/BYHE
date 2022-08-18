# Copyright ©2022 Sun weiyu and Chen ying. All Rights Reserved.
import torch
from torch import nn
from einops import rearrange

class Conv_real_face_array(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(3, 64, kernel_size=(2, 8, 8), padding=(0, 2, 2))  # 不在T维度padding
        self.Conv_Layer2 = nn.Conv3d(64, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.BN1 = nn.BatchNorm3d(64)
        self.BN2 = nn.BatchNorm3d(32)
        # self.LN1 = nn.LayerNorm([length-1, 128, 128])
        # self.LN2 = nn.LayerNorm([length-1, 64, 64])
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.ReLU(inplace=True)

    def forward(self, input):
        # input: (1, 3, 300, 131, 131) / (1, 3, 150, 131, 131)
        output = self.Conv_Layer1(input)  # (1, 64, 299, 128, 128) / (1, 64, 149, 128, 128)
        output = self.BN1(output)
        # output = self.LN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)  # (1, 64, 299, 64, 64) / (1, 64, 149, 64, 64)
        output = self.Conv_Layer2(output)  # (1, 32, 299, 64, 64) / (1, 32, 149, 64, 64)
        output = self.BN2(output)
        # output = self.LN2(output)
        output = self.Activation1(output)
        return output  # (1, 32, 299, 64, 64) / (1, 32, 149, 64, 64)


class Conv_real_face_array2(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(0, 1, 1))  # 不在T维度padding
        self.Conv_Layer2 = nn.Conv3d(64, 8, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.BN1 = nn.BatchNorm3d(64)
        self.BN2 = nn.BatchNorm3d(8)
        # self.LN1 = nn.LayerNorm([length-3, 64, 64])
        # self.LN2 = nn.LayerNorm([length-5, 32, 32])
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.ReLU(inplace=True)

    def forward(self, input):
        # input: (1, 32, 299, 64, 64) / (1, 32, 149, 64, 64)
        output = self.Conv_Layer1(input)  # (1, 64, 297, 64, 64) / (1, 64, 147, 64, 64)
        output = self.BN1(output)
        # output = self.LN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)  # (1, 64, 297, 32, 32) / (1, 64, 147, 32, 32)
        output = self.Conv_Layer2(output)  # (1, 8, 295, 32, 32) / (1, 8, 145, 32, 32)
        output = self.BN2(output)
        # output = self.LN2(output)
        output = self.Activation1(output)
        return output  # (1, 8, 295, 32, 32) / (1, 8, 145, 32, 32)


class Conv_residual_array(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(3, 32, kernel_size=(1, 8, 8), padding=(0, 2, 2))
        self.Conv_Layer2 = nn.Conv3d(32, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.BN1 = nn.BatchNorm3d(32)
        self.BN2 = nn.BatchNorm3d(16)
        # self.LN1 = nn.LayerNorm([length-1, 128, 128])
        # self.LN2 = nn.LayerNorm([length-1, 64, 64])
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.Tanh()
        self.Drop = nn.Dropout(0.15)

    def forward(self, input):
        # input: (1, 3, 299, 131, 131) / (1, 3, 149, 131, 131)
        output = self.Conv_Layer1(input)  # (1, 32, 299, 128, 128) / (1, 32, 149, 128, 128)
        output = self.BN1(output)
        # output = self.LN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)  # (1, 32, 299, 64, 64) / (1, 32, 149, 64, 64)
        output = self.Conv_Layer2(output)  # (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)
        output = self.Drop(output)
        output = self.BN2(output)
        # output = self.LN2(output)
        output = self.Activation1(output)
        return output  # (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)


class Conv_residual_array2(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.BN1 = nn.BatchNorm3d(32)
        # self.LN1 = nn.LayerNorm([length-3, 64, 64])
        self.Conv_Layer2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.BN2 = nn.BatchNorm3d(64)
        # self.LN2 = nn.LayerNorm([length-5, 32, 32])
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.Tanh()
        self.Drop = nn.Dropout(0.15)

    def forward(self, input):
        # input: (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)
        output = self.Conv_Layer1(input)  # (1, 32, 297, 64, 64) / (1, 32, 147, 64, 64)
        output = self.BN1(output)
        # output = self.LN1(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)  # (1, 32, 297, 32, 32) / (1, 32, 147, 32, 32)
        output = self.Conv_Layer2(output)  # (1, 64, 295, 32, 32) / (1, 64, 145, 32, 32)
        output = self.Drop(output)
        output = self.BN2(output)
        # output = self.LN2(output)
        output = self.Activation1(output)
        return output  # (1, 64, 295, 32, 32) / (1, 64, 145, 32, 32)


class Conv_residual_array3(nn.Module):
    def __init__(self, length=70):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(64, 32, kernel_size=(5, 3, 3), padding=(0, 0, 0))
        self.BN1 = nn.BatchNorm3d(32)
        # self.LN1 = nn.LayerNorm([length-9, 30, 30])
        self.Conv_Layer2 = nn.Conv3d(32, 8, kernel_size=(7, 3, 3), padding=(0, 0, 0))
        self.BN2 = nn.BatchNorm3d(8)
        # self.LN2 = nn.LayerNorm([length-15, 28, 28])
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Activation1 = nn.Tanh()
        self.Drop = nn.Dropout(0.15)

    def forward(self, input):
        # input: (1, 64, 295, 32, 32) / (1, 64, 145, 32, 32)
        output = self.Conv_Layer1(input)  # (1, 32, 291, 30, 30) / (1, 32, 141, 30, 30)
        output = self.BN1(output)
        # output = self.LN1(output)
        output = self.Activation1(output)
        output = self.Conv_Layer2(output)  # (1, 8, 285, 28, 28) / (1, 8, 135, 28, 28)
        output = self.Drop(output)
        output = self.BN2(output)
        # output = self.LN2(output)
        output = self.Activation1(output)
        output = self.Avgpool1(output)  # (1, 8, 285, 14, 14) / (1, 8, 135, 14, 14)
        return output  # (1, 8, 285, 14, 14) / (1, 8, 135, 14, 14)


class Projection1(nn.Module):
    def __init__(self, args, num_of_multihead=4):
        super().__init__()
        dim = 8 * args.win_length
        self.heads = num_of_multihead  # 4
        self.num_of_multihead = self.heads
        self.FCN = nn.Linear(22, 1)  # 为了兼容旧版本保存的权重
        self.to_q = nn.Linear(dim, dim)
        self.Drop = nn.Dropout(0.1)

    def forward(self, input):
        # input: (1, 275, 88), 滑动窗口生成的275个窗，每个窗为88维feature vector
        projections = self.to_q(input)  # (1, 275, 88)
        projections = self.Drop(projections)
        projection_space = projections.unsqueeze(1)  # (1, 1, 275, 88)
        attn_raw_heads = cal_cos_similarity_self(projection_space)  # (b h t t), (1, 1, 275, 275)
        attn_raw = attn_raw_heads.squeeze(1)
        return attn_raw  # (1, 275, 275)


class Split_Module(nn.Module):
    def __init__(self, slice_lens=11):
        super().__init__()
        self.slice_lens = slice_lens  # L=11

    def forward(self, input):
        # input: (1, 8, 285), 每个t是8维feature vector
        input = rearrange(input, "b c t -> b t c")  # input: (1, 285, 8)
        split_slices = []
        for i in range(input.shape[1] + 1 - self.slice_lens):  # num_slices = N-L-C+1, N=299, L=11, C=14(因没有padding损失的t)
            split_slices.append(input[:, i:i + self.slice_lens, :].unsqueeze(1))  # 滑动窗口, 重叠1个t
        split_result = torch.cat(split_slices, 1)  # 275 * (1, 1, 11, 8) → (1, 275, 11, 8)
        split_result = rearrange(split_result, "b s c t -> b s (c t)")
        return split_result  # (1, 275, 88)


def cal_cos_similarity_self(projection_space):
    output_matrix_length = projection_space.shape[2]
    # print("长度是", output_matrix_length)
    total_list = [[None for _ in range(output_matrix_length)] for _ in range(output_matrix_length)]
    result_list = []
    mod_list = [torch.sqrt(projection_space[:, :, index, :] @ rearrange(projection_space[:, :, index, :], "b o l -> b l o")) for index in range(output_matrix_length)]
    # print("mod_list", mod_list)
    for index in range(output_matrix_length):
        for index1 in range(index):
            total_list[index][index1] = total_list[index1][index]
        for index1 in range(index, output_matrix_length):
            total_list[index][index1] = (projection_space[:, :, index, :] @ rearrange(projection_space[:, :, index1, :], "b o l -> b l o")/(mod_list[index] * mod_list[index1])).squeeze(-1)
            # print(f"index1: {index} {index1}", total_list[index][index1], (torch.sqrt(projection_space[:, :, index, :] @ rearrange(projection_space[:, :, index1, :], "b o l -> b l o"))
            # total_list.append(torch.cosine_similarity(projection_space[:, :, index, :],
            #                                           projection_space[:, :, index1, :], dim=-1).unsqueeze(dim=-1))
        local_list_final = torch.cat(total_list[index], dim=-1)
        # print(local_list_final.shape) #(batch_size, [1, 275])
        result_list.append(local_list_final.unsqueeze(dim=-1))
    total_list_final = torch.cat(result_list, dim=-1)
    # print("final result", total_list_final.shape)
    return total_list_final


class super_fusion(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.c = c
        self.Linear_layer = nn.Linear(self.c, 1)
        self.Activation = nn.Sigmoid()

    def forward(self, motion_mask, appearance_mask):
        """由real face计算权重，作用在residual上"""
        # motion_mask: residual                                      appearance_mask: real face
        # motion_mask: (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)   appearance_mask: (1, 32, 299, 64, 64) / (1, 32, 149, 64, 64)
        coefficient = appearance_mask.shape[-1] * appearance_mask.shape[-2]  # 64 * 64
        appearance_compression = self.Linear_layer(rearrange(appearance_mask, "b c t h w -> b t h w c")).squeeze(-1)  # (1, 299, 64, 64) / (1, 149, 64, 64)
        appearance_compression = self.Activation(appearance_compression)
        appearance_compression = rearrange(appearance_compression, "b t h w -> (b t) h w")
        appearance_compression_slice = torch.split(appearance_compression, 1, 0)  # 299 * (1, 64, 64) / 149 * (1, 64, 64)
        appearance_compression_final = []
        for index in range(len(appearance_compression_slice)):  # 299 / 149
            motion_image_weight = coefficient * appearance_compression_slice[index] / (2 * torch.norm(appearance_compression_slice[index], 1))  # MIT论文中的（12）号公式
            appearance_compression_final.append(motion_image_weight)
        motion_image_weight = torch.cat(appearance_compression_final, 0)  # (299, 64, 64) / (149, 64, 64)
        motion_image_weight = rearrange(motion_image_weight, "(b t) h w -> b t h w", b=motion_mask.shape[0])  # (1, 299, 64, 64) / (1, 149, 64, 64)
        motion_mask = motion_mask.mul(motion_image_weight.unsqueeze(1))  # (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)
        return motion_mask, motion_image_weight  # motion_mask: (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)
                                                 # motion_image_weight: (1, 299, 64, 64) / (1, 149, 64, 64)


class Ultimate_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Conv_real_face_array = Conv_real_face_array(args.length)  # step1
        self.Conv_real_face_array2 = Conv_real_face_array2(args.length)  # step2
        self.Conv_residual_array = Conv_residual_array(args.length)
        self.Conv_residual_array2 = Conv_residual_array2(args.length)
        self.Conv_residual_array3 = Conv_residual_array3(args.length)
        self.super_fusion = super_fusion()
        self.super_fusion2 = super_fusion(8)
        self.split_module = Split_Module(args.win_length)
        # self.transformer = Transformer1()
        self.GLOBAL_AVG = nn.AvgPool3d((1, 14, 14))
        self.Projection1 = Projection1(args)

    def forward(self, input_residual, input_real_face_array):
        # input_residual: (1, 3, 299, 131, 131) / (1, 3, 149, 131, 131)
        # input_real_face_array: (1, 3, 300, 131, 131) / (1, 3, 150, 131, 131)
        output = self.Conv_real_face_array(input_real_face_array)  # (1, 32, 299, 64, 64) / (1, 32, 149, 64, 64)
        output_R = self.Conv_residual_array(input_residual)  # (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)
        output_R, face_mask = self.super_fusion(output_R, output)  # 32通道融合生成权重
        # output_R: 由real face计算出的权重作用于residual的输出, (1, 16, 299, 64, 64) / (1, 16, 149, 64, 64)
        # face_mask: 由real face计算出的权重, (1, 299, 64, 64) / (1, 149, 64, 64)

        output = self.Conv_real_face_array2(output)  # (1, 8, 295, 32, 32) / (1, 8, 145, 32, 32)
        output_R = self.Conv_residual_array2(output_R)  # (1, 64, 295, 32, 32) / (1, 64, 145, 32, 32)
        output_R1, face_mask2 = self.super_fusion2(output_R, output)  # 8通道融合生成权重
        # output_R1: (1, 64, 295, 32, 32) / (1, 64, 145, 32, 32)
        # face_mask2: (1, 295, 32, 32) / (1, 145, 32, 32)

        output_R1 = self.Conv_residual_array3(output_R1)  # (1, 8, 285, 14, 14) / (1, 8, 135, 14, 14)
        output_R = self.GLOBAL_AVG(output_R1).squeeze(-1).squeeze(-1)  # (1, 8, 285) / (1, 8, 135)

        output_R = self.split_module(output_R)  # (1, 275, 88) / (1, 125, 88)
        attn = self.Projection1(output_R)  # (1, 275, 275) / (1, 125, 125)
        return attn, face_mask
        # attn: (1, 275, 275) / (1, 125, 125)
        # face_mask: (1, 299, 64, 64) / (1, 149, 64, 64), 一阶段权重


def up_down(tensors):
    tensor_result = torch.zeros(tensors.shape)
    for i in range(tensors.shape[0]):
        tensor_result[i, :] = tensors[tensors.shape[0]-i-1, :]
    return tensor_result


class Transformer1(nn.Module):
    def __init__(self, num_of_multihead=4, dim=88, dim_head=22):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.attend2 = nn.Softmax(dim=-2)
        self.heads = num_of_multihead
        inner_dim = dim_head * self.heads
        self.num_of_multihead = self.heads
        self.L1 = nn.Linear(inner_dim, dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.FCN = nn.Linear(dim_head, 1)
        self.to_out = nn.Sequential(
            self.L1,
            nn.Dropout(0.1),
        )

    def forward(self, input):
        q = rearrange(self.to_q(input), 'b t (h d) -> b h t d', h=self.heads)
        k = rearrange(self.to_k(input), 'b t (h d) -> b h t d', h=self.heads)
        v = rearrange(self.to_v(input), 'b t (h d) -> b h t d', h=self.heads)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = torch.cat(torch.split(out, 1, 1), dim=-1).squeeze(1)
        out = self.to_out(out)
        return out


class Transformer2(nn.Module):
    def __init__(self, num_of_multihead=4, dim=88, dim_head=22):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.attend2 = nn.Softmax(dim=-2)
        self.heads = num_of_multihead
        inner_dim = dim_head * self.heads
        self.num_of_multihead = self.heads
        self.L1 = nn.Linear(inner_dim, dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.FCN = nn.Linear(dim_head, 1)
        self.to_out = nn.Sequential(
            self.L1,
            nn.Dropout(0.1),
        )

    def forward(self, input):
        q = rearrange(self.to_q(input), 'b t (h d) -> b h t d', h=self.heads)
        k = rearrange(self.to_k(input), 'b t (h d) -> b h t d', h=self.heads)
        v = rearrange(self.to_v(input), 'b t (h d) -> b h t d', h=self.heads)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)
        out = torch.matmul(attn, v)
        out = torch.cat(torch.split(out, 1, 1), dim=-1).squeeze(1)
        out = self.to_out(out)
        return out


if __name__ == "__main__":
    import numpy as np
    from torch import nn

    input = torch.FloatTensor(np.random.random((6, 3, 56, 131, 131))).cuda()
    input2 = torch.FloatTensor(np.random.random((6, 3, 57, 131, 131))).cuda()
    Ultimate_models = Ultimate_model().cuda()
    attn, mask = Ultimate_models(input, input2)
    print(attn.shape)

