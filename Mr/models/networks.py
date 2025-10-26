import torch
import torch.nn.functional as F


def Index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=False)
    return samples[:, :, :, 0]


