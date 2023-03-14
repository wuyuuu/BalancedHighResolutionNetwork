import copy
import logging

import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.cnn import (build_conv_layer, build_norm_layer)
from torch.nn.modules.batchnorm import _BatchNorm
# from .resnet import BasicBlock, Bottleneck
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .utils import InvertedResidual, load_checkpoint\
    # , DenseInvertedResidual
from .utils import SELayer



class DenseInvertedResidual(nn.Module):
    """Dense Inverted Residual Block.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution.
            Default: 3.
        groups (None or int): The group number of the depthwise convolution.
            Default: None, which means group number = mid_channels.
        stride (int): The stride of the depthwise convolution. Default: 1.
        se_cfg (dict): Config dict for se layer. Default: None, which means no
            se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels.
            Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 groups=None,
                 stride=1,
                 se_cfg=None,
                 with_expand_conv=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 attention=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv

        if groups is None:
            groups = mid_channels

        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        DCONV_NUM = 2
        dconvs = []
        for i in range(DCONV_NUM):
            # self.depthwise_conv = ConvModule(
            dconvs.append(ConvModule(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.dconvs = nn.Sequential(*dconvs)
        if self.with_se:
            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.attention = attention
        if attention:
            self.attention_block = SSA(gate_channels=in_channels)

    def forward(self, x):

        def _inner_forward(x):
            if self.attention:
                x = self.attention_block(x)
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)
            inner_shortcut = out
            out = self.dconvs(out)
            out = out + inner_shortcut
            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                return x + out
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class BalancedHRModule(nn.Module):
    def __init__(self, num_branches, in_channels, out_channels, last_layer=1, num_blocks=2, coef=2, d2_conv_num=2,
                 img_sizes=None, attention=False, attention_dir=False, base_conv_num=1):
        super(BalancedHRModule, self).__init__()
        # print(layer_settings)
        self.base_conv_num = base_conv_num
        self.exponential_dconv_num = False
        self.attention_dir = attention_dir
        self.PATCH_ATT = attention
        self.img_sizes = img_sizes
        self.coef = coef
        self.d2_conv_num = d2_conv_num
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = None
        self.norm_cfg = dict(type='BN')
        self.upsample_cfg = dict(mode='nearest', align_corners=None)

        self.num_branches = num_branches
        self.last_output_num = last_layer

        self.branches = self._make_branches()
        self.fuses = self._make_fusion()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch_block(self, in_channel, out_channel):
        """
        only make one block for one branch now.
        """
        # if self.block_type == 'Mob':
        #     block = InvertedResidual(in_channel, out_channel, self.coef * out_channel)
        block = DenseInvertedResidual(in_channel, out_channel, self.coef * out_channel, attention=self.attention_dir)
        return block

    def _make_fusion(self):
        num_branches = self.num_branches
        in_channels = self.in_channels
        out_channels = self.out_channels
        fuse_layers = []
        num_out_branches = self.last_output_num
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                out_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2 ** (j - i),
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                    upsample_cfg['align_corners'])))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv_downsamples = []
                    if self.PATCH_ATT:
                        # attention_downsample = SELayer(channels=in_channels[j])
                        # attention_downsample = CBAM(in_channels[j])
                        if self.PATCH_ATT == 'SELayer':
                            attention_downsample = SELayer(channels=in_channels[j])
                        elif self.PATCH_ATT == 'SSA':
                            attention_downsample = SSA(in_channels[j])
                        else:
                            attention_downsample = CBAM(in_channels[j])
                        conv_downsamples.append(attention_downsample)
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        out_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     out_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        out_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     out_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def _make_branches(self):
        branches = []
        for i in range(self.num_branches):
            layer = []
            for j in range(
                    self.base_conv_num * self.num_blocks ** (i) if self.exponential_dconv_num else self.num_blocks * (
                            i + 1)):
                layer.append(self._make_one_branch_block(self.in_channels[i], self.in_channels[i]))
            branches.append(nn.Sequential(*layer))
        return nn.ModuleList(branches)

    def branch_outputs(self, xs):
        for i, x in enumerate(xs):
            xs[i] = self.branches[i](x)
        return xs

    def forward(self, xs):
        for i, x in enumerate(xs):
            xs[i] = self.branches[i](x)
        x_fuse = []
        for i in range(self.last_output_num):
            y = 0
            for j in range(self.num_branches):
                if i != j:
                    y = y + self.fuses[i][j](xs[j])
                else:
                    y = y + xs[i]
            x_fuse.append(self.relu(y))
        return x_fuse


@BACKBONES.register_module()
class BalancedHRNet(BaseBackbone):
    """
    HRMob backbone
    """
    channels = ((32,),
                (32, 32),
                (32, 32, 64),
                (32, 32, 64, 128))

    def __init__(self, channels=None, num_blocks=3, coef=2, attention=False, attention_dir=False,
                 base_conv_num=1):
        super(BalancedHRNet, self).__init__()
        self.channels = channels
        self.conv_cfg = None
        self.norm_cfg = dict(type='BN')
        self.in_channels = self.channels[0][0]
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, self.in_channels // 2, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, self.in_channels, postfix=2)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels=3,
            out_channels=self.in_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            self.in_channels // 2,
            self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.in_channels = self.channels
        self.out_channels = self.channels[1:] + self.channels[:1]
        self.num_branches = list(range(1, len(self.channels) + 1))
        self.last_layers = self.num_branches[1:] + [1]
        self.stages = []
        self.img_sizes = [128, 64, 32, 16]
        for i, num_branch in enumerate(self.num_branches):
            stage = BalancedHRModule(num_branch, in_channels=channels[i], out_channels=self.out_channels[i],
                                 last_layer=self.last_layers[i],
                                 num_blocks=num_blocks, coef=coef, img_sizes=self.img_sizes, attention=attention,
                                 attention_dir=attention_dir, base_conv_num=base_conv_num)
            self.stages.append(stage)
        self.stages = nn.ModuleList(self.stages)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def stages_output(self, x, depth=[0], spec_stage=[3]):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = [x]
        outputs = {}
        for i, stage in enumerate(self.stages):
            y_list = stage(x)
            branch_outputs = stage.branch_outputs(x)
            if i in spec_stage:
                for d in depth:
                    if d in list(range(len(branch_outputs))):
                        outputs[(i, d)] = branch_outputs[d].detach()
            x = y_list
        return outputs

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = [x]
        # y_list = self.stages[0](x)
        # return y_list
        for stage in self.stages:
            y_list = stage(x)
            x = y_list
            # break
        return x


if __name__ == '__main__':
    import torch
    import time

    model = BiasHRMob().cuda()
    for i in range(10):
        x = torch.rand(size=(3, 256, 256)).cuda()
        start_time = time.time()
        y = model(x)
        print('forward time', time.time() - start_time)
