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
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
"""
Sinusoidal positional embed
"""

# Transformer model 의 Positianl Encoding 을 활용
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()    
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weight = nn.Parameter(torch.randn(half_dim), requires_grad= not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weight, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    

"""
building block modules
"""

# Resnet 내 block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8): # group 의미??
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out) # WeightStandardizedConv 를 쓴 이후에 다시 GroupNorm 을 쓰는 이유는???
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None): # scale_shift??
        x = self.proj(x)
        x = self.norm(x)

        if exist(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # time_emb 이 있을 경우
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out*2))
        else:
            self.mlp = None
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        if dim != dim_out:
            self.res_cov = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_cov = nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None

        if exist(self.mlp) and exist(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1) # chunk ??

        h = self.block1(x, scale_shift=scale_shift) # scale_shift 는 어떤 값인가?
        h = self.block2(h)

        return h + self.res_cov(x)
    

# self-attension layer
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5 # sqrt(dimension of K vector)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False) # qkv vector 를 만드는데 왜 Convolution 연산??

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    LayerNorm(dim))
        
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v) #einsum 함수 찾아보기

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    pass


"""
Model
"""
class Unet(nn.Module):
    def __init__(self,
                 dim, 
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 self_condition=False,
                 resnet_block_groups=8,
                 learned_variance=False,
                 learned_sinsoidal_cond=False,
                 random_fourier_features=False,
                 learned_sinsoidal_dim=16):
        super().__init__()
        
        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim) # init dim 이 있으면 init dim, 아니면 dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # 결과 : [init_dim, dim_mults[0] * dim, dim_mults[1] * dim, ...]
        in_out = list(zip(dims[:-1], dims[1:])) # 각 block 의 in_dimension, out_dimension 을 만든다

        block_klass = partial(ResnetBlock, groups=resnet_block_groups) # Resnet block을 그대로 새로운 함수 구현

        # time embedding
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinsoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalEmb(learned_sinsoidal_dim, random_fourier_features)
            fourier_dim = learned_sinsoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(sinu_pos_emb,
                                      nn.Linear(fourier_dim, time_dim),
                                      nn.GELU(),
                                      nn.Linear(time_dim, time_dim))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # in_out 의 길이 = unet block 의 수

        # donw blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                # PreNorm 으로 LayerNorm 한 후 LinearAttention 수행
                # Residual 로 PreNorm(dim_in) + dim_in 
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                down_sample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        # middle block
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # up block
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1) # reversed(in_out) 의 index 가 마지막 index 인지 확인하는 값

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim), # Unet 구조이므로 input_ch 수는 donw layer 의 dim_out + mid layer 의 dim_out == dim_in
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                up_sample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1) # 마지막 layer 일 때는 conv
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.fianl_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = [] # up layer 로 보낼 값

        # down
        for block1, block2, attn, domwnsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) # donw ModuleList 에서 Residual
            h.append(x)

            x = domwnsample(x)
        
        # mid
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # up
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.fianl_res_block(x)

        return self.final_conv(x)
    
"""
Gaussian Diffusion Trainer Class
"""

if __name__ == "__main__":
    pass