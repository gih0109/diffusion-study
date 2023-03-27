import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoder

from torch.optim import Adam

from torchvision import transforms as T, utils

# einops 패키지 : 행렬 차원 변환, https://yongwookha.github.io/MachineLearning/2021-10-15-einops-the-easiest-dimension-managing-lib
from einops import rearrange, reduce, repeat 
# einops.layer : pytorch layer 애서 차원 변환
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
# EMA ??
# from ema_pytorch import EMA

# accelerate : https://github.com/huggingface/accelerate
from accelerate import Accelerator

# GAN 평가 스코어 패키지 (이미지 스코어)
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

# ??
# from denoising_diffusion_pytorch.version import __version__


# constant
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

"""
help functions
"""

def exist(x):
    return x is not None

# val 이 있을경우 val 반환 아니면 d 반환
def default(val, d):
    if exist(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

# iterate로 dl 계속 순환
def cycle(dl):
    while True:
        for data in dl:
            yield data

# root num 이 정수인지 확인
def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# 나누는 수 * 몫 + 나머지 를 리스트화 -> [나누는수, 나누는수, ... , 나머지], sum(list) = num
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    # divmod 로 대체가능
    # groups, reminder = divmod(num, divisor)
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 이미지가 요구 이미지 타입과 맞지 않을 경우 변환
def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


"""
normalization functions
"""

# 값 범위를 0 ~ 1 사이로 반환
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# ??
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


"""
small helper modules
"""

# 잔차 더하기 (time embbeding val)
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    

def up_sample(dim, dim_out=None): # 원 함수명 Upsample, 함수는 snake_case 쓰기 때문에 변경
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))

def down_sample(dim, dim_out=None):
    return nn.Sequential(
        # einops Rearrange 을 이용해 레이어 shape 변경
        # h -> 1/2, w -> 1/2, c * 4 : h, w 를 반으로 줄이고 채널수 4배
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), kernel_size=1))


# Batch 수가 적을 때 Normalization 하는 방법
# WeightStandarization 은 weight(Conv filter) 를 대상으로 normalization 수행 평균=0, 분산=1 로 조정
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        # einops 의 reduce 함수 사용하여 평균 및 분산 구한다
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.val, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

if __name__ == "__main__":
    print(num_to_groups(10, 3))