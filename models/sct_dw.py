from einops import rearrange
from torch.nn import *
from mmcv.cnn import build_activation_layer, build_norm_layer
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
import numpy as np

import torch
from torch.nn import Module, ModuleList, Upsample
from mmcv.cnn import ConvModule

from torch.nn import Sequential, Conv2d, UpsamplingBilinear2d

import torch.nn as nn


class DWConv(Sequential):
    def __init__(self, dim: int):
        super(DWConv, self).__init__(
            Rearrange('b h w c -> b c h w'),
            Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
            Rearrange('b c h w -> b h w c'), )


class MLP(Sequential):
    def __init__(self, dim: int, expansion: int, act_cfg: dict = None, drop=0.):
        hid_dim = dim * expansion

        if act_cfg is None:
            act_cfg = dict(type='GELU')

        super(MLP, self).__init__(
            Linear(dim, hid_dim),
            DWConv(hid_dim),
            build_activation_layer(act_cfg),
            Linear(hid_dim, dim),
            Dropout(drop),
        )


class Attention(Module):

    def __init__(self, dim: int, num_heads: int, reduction: int, qkv_bias=False, aff_drop=0., proj_drop=0.):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim_head = dim // num_heads

        # 线性映射换成33卷积
        self.to_q = Conv2d(dim, dim, 3, 1, 1, bias=qkv_bias, groups=dim)
        self.to_sr = Conv2d(dim, dim, kernel_size=reduction, stride=reduction)
        self.to_kv = Conv2d(dim, dim * 2, 3, 1, 1, bias=qkv_bias, groups=dim)

        #        self.to_q = Linear(dim, dim, bias=qkv_bias)
        #        self.to_sr = Conv2d(dim, dim, kernel_size=reduction, stride=reduction)
        #        self.to_kv = Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = Linear(dim, dim)
        self.aff_drop = Dropout(aff_drop)
        self.proj_drop = Dropout(proj_drop)

        self.sr_norm = LayerNorm(dim)

        self.hwc2chw = Rearrange('b h w c -> b c h w')
        self.chw2hwc = Rearrange('b c h w -> b h w c')

    def forward(self, x):
    
        _, h, w, _ = x.shape
        q = self.chw2hwc(self.to_q(self.hwc2chw(x)))
        sr = self.to_sr(self.hwc2chw(x))
        sr = self.sr_norm(self.chw2hwc(sr))
        k, v = self.chw2hwc(self.to_kv(self.hwc2chw(sr))).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b h w (n c)  -> b n (h w) c', c=self.dim_head), (q, k, v))
        aff = q @ torch.transpose(k, -2, -1) * self.dim_head ** -.5
        aff = torch.softmax(aff, dim=-1)
        aff = self.aff_drop(aff)
        agg = aff @ v
        agg = rearrange(agg, 'b n (h w) c -> b h w (n c)', h=h, w=w)
        agg = self.proj(agg)
        x = self.proj_drop(agg)
        return x


class Block(Module):

    def __init__(self, dim: int, num_heads: int, reduction: int, expansion: int, qkv_bias=False,
                 drop=0., aff_drop=0., path_drop=0., act_cfg: dict = None, norm_cfg: dict = None):
        super().__init__()

        if act_cfg is None:
            act_cfg = dict(type='GELU')

        if norm_cfg is None:
            norm_cfg = dict(type='LN')

        self.attn = Attention(
            dim=dim, num_heads=num_heads, reduction=reduction, qkv_bias=qkv_bias, aff_drop=aff_drop, proj_drop=drop
        )

        self.mlp = MLP(dim=dim, expansion=expansion, act_cfg=act_cfg, drop=drop)

        self.norm1 = build_norm_layer(norm_cfg, num_features=dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, num_features=dim)[1]

        self.drop_path = DropPath(path_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(Sequential):

    def __init__(self, in_dim, out_dim, kernel_size: int, stride: int):
        super().__init__(
            Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            Rearrange('b c h w -> b h w c'),
            LayerNorm(out_dim)
        )


class MitEncoder(Module):
    def __init__(self, in_dim: int = 3,
                 dims: tuple = (64, 128, 256, 512),
                 num_heads: tuple = (1, 2, 4, 8),
                 reductions=(8, 4, 2, 1),
                 expansions: tuple = (4, 4, 4, 4),
                 depths: tuple = (3, 4, 6, 3),
                 drop=0., aff_drop=0., path_drop=0.1, qkv_bias=False,
                 act_cfg: dict = None,
                 norm_cfg: dict = None,
                 ):
        super().__init__()

        k = (7, 3, 3, 3)
        s = (4, 2, 2, 2)

        in_dims = (in_dim, *dims[:-1])

        self.embeds = ModuleList()
        self.stages = ModuleList()

        self.hwc2chw = Rearrange('b h w c -> b c h w')
        self.chw2hwc = Rearrange('b c h w -> b h w c')

        path_drops = [r for r in np.linspace(0, path_drop, sum(depths))]

        for i in range(4):
            self.embeds.append(
                PatchEmbed(in_dim=in_dims[i], out_dim=dims[i], kernel_size=k[i], stride=s[i])
            )

            self.stages.append(
                Sequential(
                    *[
                        Block(dim=dims[i], num_heads=num_heads[i], reduction=reductions[i], expansion=expansions[i],
                              qkv_bias=qkv_bias, drop=drop,
                              aff_drop=aff_drop, path_drop=path_drops[sum(depths[:i]) + j], act_cfg=act_cfg,
                              norm_cfg=norm_cfg) for j in
                        range(depths[i])
                    ]
                )
            )

    def forward(self, x):
        features = []

        for embed, stage in zip(self.embeds, self.stages):
            x = embed(x)  # [2, 128, 128, 64]
            x = stage(x)
            x = self.hwc2chw(x)
            features.append(x)

        return features


class Decoder(Module):

    def __init__(self, dims, dim, class_num=2):
        super(Decoder, self).__init__()

        self.class_num = class_num

        self.layers = ModuleList(
            [Sequential(Conv2d(dims[i], dim, 1), Upsample(scale_factor=2 ** i)) for i in range(len(dims))])

        self.conv_fuse = ConvModule(
            in_channels=dim * 4,
            out_channels=dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.conv_fuse_12 = ConvModule(
            in_channels=dim * 2,
            out_channels=dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.conv1 = ConvModule(
            in_channels=dim,
            out_channels=self.class_num,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.conv2 = ConvModule(
            in_channels=dim,
            out_channels=self.class_num,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        self.Conv2d = Conv2d(self.class_num * 2, self.class_num, 1)

    def forward(self, features):
        fuse = []

        for feature, layer in zip(features, self.layers):
            fuse.append(layer(feature))

        fused_12 = torch.cat([fuse[0], fuse[1]], dim=1)  # ([1, 1536, 128, 128])
        fused_12 = self.conv_fuse_12(fused_12)  # 2, 768, 128, 128
        fused_12 = self.conv1(fused_12)

        fuse = torch.cat(fuse, dim=1)  # [1, 3072, 128, 128]
        fuse = self.conv_fuse(fuse)  # [1, 768, 128, 128]
        fuse = self.conv2(fuse)  # [1, 768, 128, 128]
        fused = torch.cat([fuse, fused_12], dim=1)  # [1, 1536, 128, 128]
        fused = self.Conv2d(fused)

        return fused


class sct_b0(nn.Module):
    def __init__(self,class_num=2, **kwargs):
    
        super(sct_b0, self).__init__()
        self.class_num = class_num
        self.backbone = MitEncoder(in_dim=3, dims=(32, 64, 160, 256),
                                   num_heads=(1, 2, 5, 8), expansions=(4, 4, 4, 4),
                                   reductions=(8, 4, 2, 1), depths=(2, 2, 2, 2), qkv_bias=True, drop=0, aff_drop=0,
                                   path_drop=.1, norm_cfg=dict(type='LN', eps=1e-6), act_cfg=dict(type='GELU'))

        self.decode_head = Decoder(dims=[32, 64, 160, 256], dim=256, class_num=class_num)
    def forward(self, x):
        features = self.backbone(x)

        features = self.decode_head(features)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(features)
        print(features.shape)
        return features


class sct_b1(nn.Module):
    def __init__(self,class_num=2, **kwargs):
    
        super(sct_b1, self).__init__()
        self.class_num = class_num
        self.backbone = MitEncoder(in_dim=3, dims=(64, 128, 320, 512),
                                   num_heads=(1, 2, 5, 8), expansions=(4, 4, 4, 4),
                                   reductions=(8, 4, 2, 1), depths=(8, 4, 2, 2), qkv_bias=True, drop=0, aff_drop=0,
                                   path_drop=.1, norm_cfg=dict(type='LN', eps=1e-6), act_cfg=dict(type='GELU'))

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=512, class_num=class_num)
    def forward(self, x):
        features = self.backbone(x)

        features = self.decode_head(features)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(features)
        print(features.shape)
        return features


class sct_b2(nn.Module):
    def __init__(self,class_num=2, **kwargs):
    
        super(sct_b2, self).__init__()
        self.class_num = class_num
        self.backbone = MitEncoder(in_dim=3, dims=(64, 128, 320, 512),
                                   num_heads=(1, 2, 5, 8), expansions=(4, 4, 4, 4),
                                   reductions=(8, 4, 2, 1), depths=(3, 4, 6, 3), qkv_bias=True, drop=0, aff_drop=0,
                                   path_drop=.1, norm_cfg=dict(type='LN', eps=1e-6), act_cfg=dict(type='GELU'))

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=768, class_num=class_num)
    def forward(self, x):
        features = self.backbone(x)

        features = self.decode_head(features)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(features)
        print(features.shape)
        return features

class sct_b3(nn.Module):
    def __init__(self,class_num=2, **kwargs):
    
        super(sct_b3, self).__init__()
        self.class_num = class_num
        self.backbone = MitEncoder(in_dim=3, dims=(64, 128, 320, 512),
                                   num_heads=(1, 2, 5, 8), expansions=(4, 4, 4, 4),
                                   reductions=(8, 4, 2, 1), depths=(3, 4, 18, 3), qkv_bias=True, drop=0, aff_drop=0,
                                   path_drop=.1, norm_cfg=dict(type='LN', eps=1e-6), act_cfg=dict(type='GELU'))

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=768, class_num=class_num)
    def forward(self, x):
        features = self.backbone(x)

        features = self.decode_head(features)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(features)
        print(features.shape)
        return features

class sct_b4(nn.Module):
    def __init__(self,class_num=2, **kwargs):
    
        super(sct_b4, self).__init__()
        self.class_num = class_num
        self.backbone = MitEncoder(in_dim=3, dims=(64, 128, 320, 512),
                                   num_heads=(1, 2, 5, 8), expansions=(4, 4, 4, 4),
                                   reductions=(8, 4, 2, 1), depths=(3, 8, 27, 3), qkv_bias=True, drop=0, aff_drop=0,
                                   path_drop=.1, norm_cfg=dict(type='LN', eps=1e-6), act_cfg=dict(type='GELU'))

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=768, class_num=class_num)
    def forward(self, x):
        features = self.backbone(x)

        features = self.decode_head(features)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(features)
        print(features.shape)
        return features

class sct_b5(nn.Module):
    def __init__(self,class_num=2, **kwargs):
    
        super(sct_b5, self).__init__()
        self.class_num = class_num
        self.backbone = MitEncoder(in_dim=3, dims=(64, 128, 320, 512),
                                   num_heads=(1, 2, 5, 8), expansions=(4, 4, 4, 4),
                                   reductions=(8, 4, 2, 1), depths=(3, 6, 40, 3), qkv_bias=True, drop=0, aff_drop=0,
                                   path_drop=.1, norm_cfg=dict(type='LN', eps=1e-6), act_cfg=dict(type='GELU'))

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=768, class_num=class_num)
    def forward(self, x):
        features = self.backbone(x)

        features = self.decode_head(features)
        up = UpsamplingBilinear2d(scale_factor=4)
        features = up(features)

        return features
               
# MitEncoder = sct_b4(class_num=2)
# from torchinfo import summary

# summary = summary(MitEncoder, (8, 3, 512, 512))


