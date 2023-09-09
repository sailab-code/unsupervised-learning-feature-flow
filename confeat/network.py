import copy
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms, datasets, models
from torchvision.models.resnet import conv3x3

from confeat.pwc_utils.warp_utils import get_occu_mask_backward, get_occu_mask_bidirection
from confeat.utils import recursive_avoid_bn, recursive_avoid_maxpool, \
    recursive_reduce_num_filters, fix_first_layer
from lve.utils import backward_warp, sampled_similarity_dissimilarity_loss
from dpt_intel.dpt.models import DPTSegmentationModel
import torchvision
from torchvision.transforms import Compose
import os
from dpt_intel.dpt.transforms import Resize, NormalizeImage, PrepareForNet

def squash(x, gamma, binary=False):
    if binary:
        return (x > gamma).to(x.dtype)
    else:
        return 1 - torch.exp(-gamma * x)


def recursive_computes(base_model, factor=2):
    for id, (name, child_model) in enumerate(base_model.named_children()):
        if isinstance(child_model, nn.Conv2d):
            orig_conv = child_model
            inplanes = int(orig_conv.in_channels // factor)
            if inplanes == 0: inplanes = 1
            new_conv = torch.nn.Conv2d(in_channels=inplanes,
                                       out_channels=int(orig_conv.out_channels // factor),
                                       kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
                                       groups=orig_conv.groups,
                                       dilation=orig_conv.dilation, padding_mode=orig_conv.padding_mode,
                                       padding=orig_conv.padding, bias=orig_conv.bias)
            setattr(base_model, name, new_conv)

    for name, immediate_child_module in base_model.named_children():
        recursive_reduce_num_filters(immediate_child_module, factor)


def compute_photometric_loss(f1, f2, charb_eps, charb_alpha, loss_type="charb", occlusion_mask=None, moving_mask=None):
    if loss_type == "charb":
        photo_2d = ((f1 - f2) ** 2 + charb_eps ** 2).pow(charb_alpha) - math.pow(charb_eps ** 2, charb_alpha)
        if occlusion_mask is not None:
            photo_2d = photo_2d * occlusion_mask
        if moving_mask is not None:
            f = f1.shape[1]
            photo_2d = photo_2d * moving_mask  # [b, f, h, w]
            num_active_pixels = torch.sum(moving_mask, dim=(2, 3), keepdim=True)
            # sum over dims f, h, w -> divide by f * num_active_pixels
            sample_mean = torch.sum(photo_2d,  dim=(1, 2, 3), keepdim=True) / (f * num_active_pixels + 1e-10)
            overall_mean = sample_mean.mean()  # mean over all the batch
            return overall_mean, photo_2d
        return photo_2d.mean(), photo_2d
    elif loss_type == "kl_div":
        # https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net
        log_f1 = torch.log(f1 + 1e-20)
        log_f2 = torch.log(f2 + 1e-20)
        if occlusion_mask is not None:
            log_f1 = log_f1 * occlusion_mask
            log_f2 = log_f2 * occlusion_mask
        if moving_mask is not None:
            raise NotImplementedError
            # log_f1 = log_f1 * moving_mask
            # log_f2 = log_f2 * moving_mask
        return F.kl_div(log_f1, log_f2, log_target=True, reduction='mean') + F.kl_div(log_f2, log_f1, log_target=True,
                                                                                      reduction='mean'), None
    elif loss_type == "xent":
        photo_2d = f2 * torch.log(f1 + 1e-20) + f1 * torch.log(f2 + 1e-20)
        if occlusion_mask is not None: photo_2d = photo_2d * occlusion_mask
        if moving_mask is not None:
            photo_2d = photo_2d * moving_mask
            f = f1.shape[1]
            num_active_pixels = torch.sum(moving_mask, dim=(2, 3), keepdim=True)
            sample_mean = torch.sum(photo_2d, dim=(1, 2, 3), keepdim=True) / (f * num_active_pixels + 1e-10)
            overall_mean = - sample_mean.mean()  # mean over all the batch
            return overall_mean, photo_2d
        else:
            return torch.mean(-torch.mean(photo_2d, 1)), photo_2d
    else:
        raise NotImplementedError


def compute_reconstruction_loss(f1, f2, charb_eps, charb_alpha, decoder_loss):
    if decoder_loss == "l1":
        loss = F.l1_loss(f1, f2)
    elif decoder_loss == "mse":
        loss = F.mse_loss(f1, f2)
    elif decoder_loss == "photometric":
        loss, _ = compute_photometric_loss(f1, f2, charb_eps, charb_alpha)
    else:
        raise NotImplementedError
    return loss


def compute_mi_approx_loss(p, p_avg=None, lambdas=None, mask=None):
    """The function that computes the value the (approximated) mutual information.

    Args:
        p (tensor): tensor of pm pixel-wise probabilities, b x pm x h x w
        p_avg (tensor): tensor of pm average pixel-wise probabilities, b x pm x h x w
        lambdas (tuple of two float): weight of the conditional entropy and entropy terms, respectively.
        mask (tensor): b x 1 x h x w with elements in [0,1], weighing the pixel importance

    Returns:
        Weighed mutual information (ready to be maximized).
        Conditional entropy.
        Entropy.
    """
    if mask is None:
        cond_entropy = -torch.sum(torch.pow(p, 2)).div(p.shape[0] * p.shape[2] * p.shape[3])
        p_avg = torch.mean(p, dim=(0, 2, 3)) if p_avg is None else p_avg
        entropy = -torch.sum(torch.pow(p_avg, 2))
    else:
        mask = mask / torch.sum(mask)
        cond_entropy = -torch.sum(mask * torch.pow(p, 2))
        p_avg = torch.sum(mask * p, dim=(0, 2, 3)) if p_avg is None else p_avg
        entropy = -torch.sum(torch.pow(p_avg, 2))

    if lambdas is not None:
        weighed_mi = lambdas[1] * entropy - lambdas[0] * cond_entropy
    else:
        weighed_mi = entropy - cond_entropy

    return weighed_mi, cond_entropy, entropy


def compute_regularization_loss(displacements):
    return torch.mean(torch.stack([torch.mean(displacement ** 2) for displacement in displacements]))


def compute_motion_mask(flow, threshold=0.5):
    mask = torch.norm(flow, p=float('inf'), dim=1, keepdim=True) > threshold
    return mask


def compute_reconstruction_acc(warped_frame, old_frame, motion_mask, thresholds):
    if not warped_frame.shape[0] == 1: raise AssertionError
    diff = torch.norm(old_frame - warped_frame, p=float('inf'), dim=1, keepdim=True)
    diff_masked = diff[motion_mask]
    accs = {threshold:
                1.0 if torch.numel(diff_masked) == 0 else
                torch.mean((diff_masked < threshold).float()).item()
            for threshold in thresholds}

    return diff, accs


class BlockFactory:
    @staticmethod
    def createBlock(block_name, options, inplanes, planes, stride):
        if block_name == "identityblock":
            return IdentityBlock(inplanes=inplanes, planes=planes,
                                 stride=stride)
        elif block_name == "pwcblock":
            from confeat.pwclite import PwcBlock
            return PwcBlock(inplanes=inplanes // 2)
        elif block_name == "resunetblock":
            return ResUnet4Block(inplanes=inplanes, planes=planes, batch_norm=options["batch_norm"])
        elif block_name == "resunetblock_bias":
            return ResUnet4Block(inplanes=inplanes, planes=planes, bias=True, batch_norm=options["batch_norm"])
        elif block_name == "resunetnnblock_bias":
            return ResUnet4Block(inplanes=inplanes, planes=planes, bias=True, batch_norm=options["batch_norm"],
                                        upsampling='nearest')
        elif block_name == "resunetctblock":
            return ResUnet4Block(inplanes=inplanes, planes=planes, upsampling='convtranspose',
                                 batch_norm=options["batch_norm"])
        elif block_name == "resunetnnblock":
            return ResUnet4Block(inplanes=inplanes, planes=planes, upsampling='nearest',
                                 batch_norm=options["batch_norm"])
        elif block_name == "resunetblockoriginal":
            return ResUnetBlock(inplanes=inplanes, planes=planes, batch_norm=options["batch_norm"])
        elif block_name == "resunetblockoriginal_bias":
            return ResUnetBlock(inplanes=inplanes, planes=planes, bias=True, batch_norm=options["batch_norm"])
        elif block_name == "resunetnnblockoriginal":
            return ResUnetBlock(inplanes=inplanes, planes=planes, upsampling='nearest',
                                batch_norm=options["batch_norm"])
        elif block_name == "resunetctblockoriginal":
            return ResUnetBlock(inplanes=inplanes, planes=planes, upsampling='convtranspose',
                                batch_norm=options["batch_norm"])
        elif block_name == "resunetblocknoskip":
            return ResUnetBlockNoSkip(inplanes=inplanes, planes=planes, batch_norm=options["batch_norm"])
        elif block_name == "resunetblocknoskip_bias":
            return ResUnetBlockNoSkip(inplanes=inplanes, planes=planes, batch_norm=options["batch_norm"], bias=True)
        elif block_name == "resunetnnblocknoskip":
            return ResUnetBlockNoSkip(inplanes=inplanes, planes=planes, upsampling='nearest',
                                      batch_norm=options["batch_norm"])
        elif block_name == "resunetctblocknoskip":
            return ResUnetBlockNoSkip(inplanes=inplanes, planes=planes, upsampling='convtranspose',
                                      batch_norm=options["batch_norm"])
        elif block_name == "resunetblocknolastskip":
            return ResUnetBlockNoLastSkip(inplanes=inplanes, planes=planes, batch_norm=options["batch_norm"])
        elif block_name == "resunetblocknolastskip_bias":
            return ResUnetBlockNoLastSkip(inplanes=inplanes, planes=planes, batch_norm=options["batch_norm"], bias=True)
        elif block_name == "dpt_backbone":
            return DPTBackboneBlock(inplanes=inplanes, planes=planes, stride=stride,
                                    evaluate_only=options["evaluate_only"])
        elif block_name == "dpt_classifier":
            return DPTClassifierBlock(inplanes=inplanes, planes=planes, stride=stride,
                                      evaluate_only=options["evaluate_only"])
        elif block_name == "from_disk":
            return FromDisk(inplanes=inplanes, planes=planes, stride=stride)
        else:
            raise NotImplementedError


class ConvBlockBN(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
    ) -> None:
        super().__init__()

        norm_layer = nn.BatchNorm2d

        int_planes = planes // 2
        self.conv1 = conv3x3(inplanes, int_planes, stride)
        self.bn1 = norm_layer(int_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(int_planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x, y=None, z=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)



class FromDisk(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
    ) -> None:
        super().__init__()

    def forward(self, x):
        """
        Optional code for smoothing etc
        :param x:
        :return:
        """
        return x


class IdentityBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
    ) -> None:
        super().__init__()

    def forward(self, x):
        return x


def convrelu(in_channels, out_channels, kernel, padding, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
    )


class ResNetUNet4(nn.Module):

    def __init__(self, n_out, batch_norm, c=3, maxpool=True, upsampling='bilinear', bias=False):
        super().__init__()
        base_model = models.resnet18(pretrained=False)
        if not batch_norm:
            recursive_avoid_bn(base_model)
        if not maxpool:
            recursive_avoid_maxpool(base_model)

        self.upsampling = upsampling
        factor = 4

        recursive_reduce_num_filters(base_model, factor)

        self.c = c
        self.base_layers = list(base_model.children())
        self.base_layers = fix_first_layer(self.base_layers, c)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64 // factor, 64 // factor, 1, 0, bias=bias)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64 // factor, 64 // factor, 1, 0, bias=bias)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128 // factor, 128 // factor, 1, 0, bias=bias)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256 // factor, 256 // factor, 1, 0, bias=bias)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512 // factor, 512 // factor, 1, 0, bias=bias)

        self.upsample_a = create_upsample(upsampling=self.upsampling, in_channels=512 // 4, out_channels=512 // 4)
        self.upsample_b = create_upsample(upsampling=self.upsampling, in_channels=512 // 4, out_channels=512 // 4)
        self.upsample_c = create_upsample(upsampling=self.upsampling, in_channels=256 // 4, out_channels=256 // 4)
        self.upsample_d = create_upsample(upsampling=self.upsampling, in_channels=256 // 4, out_channels=256 // 4)
        self.upsample_e = create_upsample(upsampling=self.upsampling, in_channels=128 // 4, out_channels=128 // 4)

        # self.upsample = nn.Upsample(scale_factor=2, mode=upsampling, align_corners=True)

        self.conv_up3 = convrelu((256 + 512) // factor, 512 // factor, 3, 1, bias=bias)
        self.conv_up2 = convrelu((128 + 512) // factor, 256 // factor, 3, 1, bias=bias)
        self.conv_up1 = convrelu((64 + 256) // factor, 256 // factor, 3, 1, bias=bias)
        self.conv_up0 = convrelu((64 + 256) // factor, 128 // factor, 3, 1, bias=bias)

        self.conv_original_size0 = convrelu(c, 64 // factor, 3, 1, bias=bias)
        self.conv_original_size1 = convrelu(64 // factor, 64 // factor, 3, 1, bias=bias)
        self.conv_original_size2 = convrelu((64 + 128) // factor, 64 // factor, 3, 1, bias=bias)

        self.conv_last = nn.Conv2d(64 // factor, n_out, 1, bias=bias)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1 and self.c == 3:
            input = input.repeat(1, 3, 1, 1)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample_a(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample_b(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample_c(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample_d(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample_e(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResUnet4Block(nn.Module):
    def __init__(self, inplanes: int,
                 planes: int, upsampling: str = 'bilinear',
                 stride: int = 1,
                 bias: bool = False,
                 batch_norm: bool = False):
        super(ResUnet4Block, self).__init__()
        self.layers = ResNetUNet4(n_out=planes, batch_norm=batch_norm, c=inplanes,
                                  maxpool=False, upsampling=upsampling, bias=bias)

    def forward(self, x):
        return self.layers(x)


def fix_channels(orig_conv, in_ch, out_ch):
    return torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                           kernel_size=orig_conv.kernel_size, stride=orig_conv.stride, groups=orig_conv.groups,
                           dilation=orig_conv.dilation, padding_mode=orig_conv.padding_mode,
                           padding=orig_conv.padding, bias=orig_conv.bias)



def create_upsample(upsampling, in_channels, out_channels):
    if upsampling == 'convtranspose':
        return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
    else:
        params = {'scale_factor': 2, 'mode': upsampling}
        if 'linear' in upsampling:
            params['align_corners'] = True
        return nn.Upsample(**params)


class ResNetUNet(nn.Module):

    def __init__(self, n_out, batch_norm, c=3, maxpool=True, upsampling='bilinear', bias=False):
        super().__init__()
        base_model = models.resnet18(pretrained=False)
        if not batch_norm:
            recursive_avoid_bn(base_model)
        if not maxpool:
            recursive_avoid_maxpool(base_model)

        self.upsampling = upsampling

        self.c = c
        self.base_layers = list(base_model.children())
        self.base_layers = fix_first_layer(self.base_layers, c)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0, bias=bias)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0, bias=bias)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0, bias=bias)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0, bias=bias)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0, bias=bias)

        self.upsample_a = create_upsample(upsampling=self.upsampling, in_channels=512, out_channels=512)
        self.upsample_b = create_upsample(upsampling=self.upsampling, in_channels=512, out_channels=512)
        self.upsample_c = create_upsample(upsampling=self.upsampling, in_channels=256, out_channels=256)
        self.upsample_d = create_upsample(upsampling=self.upsampling, in_channels=256, out_channels=256)
        self.upsample_e = create_upsample(upsampling=self.upsampling, in_channels=128, out_channels=128)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1, bias=bias)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1, bias=bias)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1, bias=bias)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1, bias=bias)

        self.conv_original_size0 = convrelu(c, 64, 3, 1, bias=bias)
        self.conv_original_size1 = convrelu(64, 64, 3, 1, bias=bias)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1, bias=bias)

        self.conv_last = nn.Conv2d(64, n_out, 1, bias=bias)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1 and self.c == 3:
            input = input.repeat(1, 3, 1, 1)

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample_a(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample_b(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample_c(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample_d(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample_e(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResUnetBlock(nn.Module):
    def __init__(self, inplanes: int,
                 planes: int, upsampling: str = 'bilinear',
                 stride: int = 1,
                 bias: bool = False,
                 batch_norm: bool = False):
        super(ResUnetBlock, self).__init__()
        self.layers = ResNetUNet(n_out=planes, batch_norm=batch_norm, c=inplanes,
                                 maxpool=False, upsampling=upsampling, bias=bias)

    def forward(self, x):
        return self.layers(x)


class ResNetUNet4NoSKip(nn.Module):
    def __init__(self, n_out, batch_norm, c=3, maxpool=True, upsampling='bilinear', bias=False):
        super(ResNetUNet4NoSKip, self).__init__()

        base_model = models.resnet18(pretrained=False)
        if not batch_norm:
            recursive_avoid_bn(base_model)
        if not maxpool:
            recursive_avoid_maxpool(base_model)

        factor = 4

        recursive_reduce_num_filters(base_model, factor)

        self.c = c
        self.upsampling = upsampling
        self.base_layers = list(base_model.children())
        self.base_layers = fix_first_layer(self.base_layers, c)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64 // factor, 64 // factor, 1, 0, bias=bias)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128 // factor, 128 // factor, 1, 0, bias=bias)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256 // factor, 256 // factor, 1, 0, bias=bias)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512 // factor, 512 // factor, 1, 0, bias=bias)

        self.conv_up3 = convrelu(512 // factor, 512 // factor, 3, 1, bias=bias)
        self.conv_up2 = convrelu(512 // factor, 256 // factor, 3, 1, bias=bias)
        self.conv_up1 = convrelu(256 // factor, 256 // factor, 3, 1, bias=bias)
        self.conv_up0 = convrelu(256 // factor, 128 // factor, 3, 1, bias=bias)

        self.conv_original_size0 = convrelu(self.c, 64 // factor, 3, 1, bias=bias)
        self.conv_original_size1 = convrelu(64 // factor, 64 // factor, 3, 1, bias=bias)
        self.conv_original_size2 = convrelu(128 // factor, 64 // factor, 3, 1, bias=bias)
        self.conv_original_size2 = convrelu(128 // factor, 64 // factor, 3, 1, bias=bias)

        self.upsample_a = create_upsample(upsampling=self.upsampling, in_channels=512 // 4, out_channels=512 // 4)
        self.upsample_b = create_upsample(upsampling=self.upsampling, in_channels=512 // 4, out_channels=512 // 4)
        self.upsample_c = create_upsample(upsampling=self.upsampling, in_channels=256 // 4, out_channels=256 // 4)
        self.upsample_d = create_upsample(upsampling=self.upsampling, in_channels=256 // 4, out_channels=256 // 4)
        self.upsample_e = create_upsample(upsampling=self.upsampling, in_channels=128 // 4, out_channels=128 // 4)

        self.conv_last = nn.Conv2d(64 // factor, n_out, 1, bias=bias)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1 and self.c == 3:
            input = input.repeat(1, 3, 1, 1)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample_a(layer4)
        x = self.conv_up3(x)

        x = self.upsample_b(x)
        x = self.conv_up2(x)

        x = self.upsample_c(x)
        x = self.conv_up1(x)

        x = self.upsample_d(x)
        x = self.conv_up0(x)

        x = self.upsample_e(x)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResUnetBlockNoSkip(nn.Module):
    def __init__(self, inplanes: int,
                 planes: int, upsampling: str = 'bilinear',
                 stride: int = 1, batch_norm: bool = False, bias: bool=False):
        super(ResUnetBlockNoSkip, self).__init__()
        self.layers = ResNetUNet4NoSKip(n_out=planes, batch_norm=batch_norm, c=inplanes,
                                        maxpool=False, upsampling=upsampling, bias=bias)

    def forward(self, x):
        return self.layers(x)


class ResNetUNet4NoLastSKip(ResNetUNet4):
    def __init__(self, n_out, batch_norm, c=3, maxpool=True, upsampling='bilinear', bias=False):
        super(ResNetUNet4NoLastSKip, self).__init__(n_out, batch_norm, c=3, maxpool=True, upsampling=upsampling, bias=bias)


        self.upsampling = upsampling
        base_model = models.resnet18(pretrained=False)
        if not batch_norm:
            recursive_avoid_bn(base_model)
        if not maxpool:
            recursive_avoid_maxpool(base_model)

        factor = 4

        recursive_reduce_num_filters(base_model, factor)

        self.c = c
        self.base_layers = list(base_model.children())
        self.base_layers = fix_first_layer(self.base_layers, c)

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64 // factor, 64 // factor, 1, 0, bias=bias)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)

        self.conv_original_size2 = convrelu(128 // factor, 64 // factor, 3, 1, bias=bias)

    def forward(self, input):
        # repeat grey channel for rgb
        if input.shape[1] == 1 and self.c == 3:
            input = input.repeat(1, 3, 1, 1)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample_a(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample_b(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample_c(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample_d(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample_e(x)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResUnetBlockNoLastSkip(nn.Module):
    def __init__(self, inplanes: int,
                 planes: int,
                 stride: int = 1,
                 batch_norm: bool=False,
                 bias: bool=False):
        super(ResUnetBlockNoLastSkip, self).__init__()
        self.layers = ResNetUNet4NoLastSKip(n_out=planes, batch_norm=batch_norm, c=inplanes,
                                            maxpool=False, bias=bias)

    def forward(self, x):
        return self.layers(x)


class StandardFBlock(nn.Module):
    def __init__(self, options, block_index):
        super().__init__()
        self.block_index = block_index
        self.block_name = options['vision_block']['features']['block_name'][block_index]

        inplanes = options['vision_block']['features']['planes'][block_index]
        if options['vision_block']['features']['use_initial_for_features']:
            inplanes += options['c']
        planes = options['vision_block']['features']['planes'][block_index + 1]

        self.block = BlockFactory.createBlock(self.block_name, options, inplanes,
                                              planes, stride=options["vision_block"]["features"]["stride"])
        self.normalize = options['vision_block']['features']['normalize']

    def forward(self, frame):
        out = self.block(frame)
        return out


default_models = {
    "dpt_large": "dpt_large-ade20k-b12dca68.pt",
    "dpt_hybrid": "dpt_hybrid-ade20k-53898607.pt",
}

model_folder = os.path.join("dpt_intel", "weights")

dpt_transform = Compose(
    [
        torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


class DPTHybridEncoderBackbone(nn.Module):
    def __init__(self, out_dim, pretrain_flag):
        super(DPTHybridEncoderBackbone, self).__init__()
        path_weights = os.path.join(model_folder, default_models["dpt_hybrid"]) if pretrain_flag else None
        self.model = DPTSegmentationModel(
            150,
            path=path_weights,
            backbone="vitb_rn50_384",
            features=out_dim
        )
        if path_weights is not None:
            self.model.eval()
        self.model.scratch.output_conv = nn.Identity()

    def forward(self, x):
        # unnormalization
        # normalization (customizable)
        # frame = (frame - 0.5) / 0.25
        input_shapes = x.shape
        x = (x * 0.25) + 0.5

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = dpt_transform(x)
        # x = F.interpolate(x, size=(net_w, net_h), mode='bilinear', align_corners=False)

        out = self.model(x)

        return F.interpolate(out, size=(input_shapes[-2], input_shapes[-1]),
                             mode='bilinear', align_corners=False)


class DPTBackboneBlock(nn.Module):
    def __init__(self, inplanes: int,
                 planes: int,
                 stride: int = 1, evaluate_only=False):
        super(DPTBackboneBlock, self).__init__()
        self.layers = DPTHybridEncoderBackbone(out_dim=planes, pretrain_flag=evaluate_only)

    def forward(self, x):
        return self.layers(x)


class DPTHybridEncoderClassifier(nn.Module):
    def __init__(self, out_dim, pretrain_flag):
        super(DPTHybridEncoderClassifier, self).__init__()
        path_weights = os.path.join(model_folder, default_models["dpt_hybrid"]) if pretrain_flag else None
        self.model = DPTSegmentationModel(
            150,
            path=path_weights,
            backbone="vitb_rn50_384",
            features=out_dim
        )
        if path_weights is not None:
            self.model.eval()
        self.model.scratch.output_conv[3], self.model.scratch.output_conv[4], self.model.scratch.output_conv[5] = \
            nn.Identity(), nn.Identity(), nn.Identity()

    def forward(self, x):
        input_shapes = x.shape
        # unnormalization
        # normalization (customizable)
        # frame = (frame - 0.5) / 0.25
        x = (x * 0.25) + 0.5

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = dpt_transform(x)
        # x = F.interpolate(x, size=(net_w, net_h), mode='bilinear', align_corners=False)

        out = self.model(x)
        return F.interpolate(out, size=(input_shapes[-2], input_shapes[-1]),
                             mode='bilinear', align_corners=False)


class DPTClassifierBlock(nn.Module):
    def __init__(self, inplanes: int,
                 planes: int,
                 stride: int = 1, evaluate_only=False):
        super(DPTClassifierBlock, self).__init__()
        self.layers = DPTHybridEncoderClassifier(out_dim=planes, pretrain_flag=evaluate_only)

    def forward(self, x):
        return self.layers(x)


class FromDiskDBlock(nn.Module):
    def __init__(self, options, block_index):
        super().__init__()

        self.block_index = block_index
        # Assuming implicit input mode
        inplanes = 0
        planes = 0
        self.block = BlockFactory.createBlock("from_disk", options, inplanes,
                                              planes, stride=0)

    def forward(self, motion_farneback):
        return {'fwd': self.block(motion_farneback)}


class StandardDBlock(nn.Module):
    def __init__(self, options, block_index):
        super().__init__()

        self.block_index = block_index
        self.options = options
        self.block_name = options['vision_block']['displacements']['block_name'][block_index]
        # Assuming implicit input mode
        options_displacements = options['vision_block']['displacements']
        inplanes = options['vision_block']['features']['planes'][block_index] * 2

        planes = options['vision_block']['displacements']['planes']
        self.block = BlockFactory.createBlock(self.block_name, options, inplanes,
                                              planes, stride=options["vision_block"]["displacements"]["stride"])

    def forward(self, current_frame, old_frame):
        if self.options['vision_block']['displacements']['occlusion'] is not None:
            return {'fwd': self.block(torch.cat([current_frame, old_frame], 1)),
                    'bwd': self.block(torch.cat([old_frame, current_frame], 1))}
        else:
            return {'fwd': self.block(torch.cat([current_frame, old_frame], 1))}


class VisionBlock(nn.Module):
    def __init__(self, options, block_index):
        super().__init__()

        self.features_block = StandardFBlock(options, block_index=block_index)
        if "motion_" in options["vision_block"]["displacements"]["block_name"][block_index]:
            self.displacements_block = FromDiskDBlock(options, block_index=block_index)
        else:
            self.displacements_block = StandardDBlock(options, block_index=block_index)
        self.feature_detach = options["vision_block"]["displacements"]["feature_detach"]

        self.features_block_teacher = None
        self.teacher_ema_weight = None
        if options['teacher']:
            self.teacher_ema_weight = options['teacher_ema_weight']
            self.features_block_teacher = copy.deepcopy(self.features_block)
        else:
            self.features_block_teacher = self.features_block

    def update_teacher_features_block(self):
        m = self.teacher_ema_weight
        assert 0. <= m <= 1.
        with torch.no_grad():
            for pt, ps in zip(self.features_block_teacher.parameters(), self.features_block.parameters()):
                pt.copy_(m * pt + (1. - m) * ps)

    def forward(self, lower_current_features, lower_old_features, motion_disk=None):
        current_features = self.features_block_teacher(lower_current_features)
        old_features = self.features_block(lower_old_features)

        if hasattr(self.features_block, 'logits'):
            current_logits = self.features_block.logits
            old_logits = self.features_block.logits
        else:
            current_logits = current_features
            old_logits = old_features

        if isinstance(self.displacements_block, FromDiskDBlock):
            displacements = self.displacements_block(motion_disk)
        else:
            # notice that displacements are computed from features coming from lower block!
            if self.feature_detach:
                # detach the features ingoing into motion prediction
                detached_lower_current_features = lower_current_features.detach()
                detached_lower_old_features = lower_old_features.detach()
                displacements = self.displacements_block(detached_lower_current_features, detached_lower_old_features)
            else:
                displacements = self.displacements_block(lower_current_features, lower_old_features)
        return (current_features, old_features), (lower_current_features, lower_old_features), displacements, \
               (current_logits, old_logits)


class BasicDecoder(nn.Module):
    def __init__(self, options, block_index):
        super(BasicDecoder, self).__init__()

        self.options = options
        self.warping_type = options['decoder']['warping_type']
        self.block_index = block_index

        additional_inplanes = 2 if self.warping_type == 'implicit' else 0

        inplanes = options['vision_block']['features']['planes'][block_index + 1] + additional_inplanes
        if options['decoder_input'] == 'concatenated':
            inplanes = sum(options['vision_block']['features']['planes'][1:]) + additional_inplanes
        planes = options['c']

        self.decoder_net = BlockFactory.createBlock(options["decoder"]["block_name"], options, inplanes,
                                                    planes, stride=options["decoder"]["stride"])

    def forward(self, current_feat, displacement):
        if self.warping_type == 'implicit':
            return self.decoder_net(torch.cat([current_feat, displacement], 1))
        else:
            est_feat = backward_warp(frame=current_feat, displacement=displacement)
            return self.decoder_net(est_feat)


class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        self.options = options
        if self.options['decoder_input'] == 'per-block':
            self.decoders = nn.ModuleList()
            for i in range(self.options['n_blocks']):
                self.decoders.append(BasicDecoder(options, block_index=i))
        elif self.options['decoder_input'] == 'last-block' or self.options['decoder_input'] == 'concatenated':
            self.decoders = nn.ModuleList([BasicDecoder(options, block_index=self.options['n_blocks'] - 1)])
        else:
            raise NotImplementedError

    def forward(self, current_features, displacements):
        outputs = []
        for decoder, current_feat, displacement in zip(self.decoders, current_features,
                                                       displacements):  # old_features, displacements
            outputs.append(decoder(current_feat, displacement))
        return outputs


class WholeNet(nn.Module):

    def __init__(self, options, device, worker=None):
        super(WholeNet, self).__init__()

        self.eps = 0.000000001

        assert options['n_blocks'] == len(
            options['vision_block']['features']['planes']) - 1, 'incompatible number of vision blocks and planes'
        # keeping track of the network options
        self.options = options
        self.device = device
        self.vision_blocks = torch.nn.ModuleList()

        for i in range(self.options['n_blocks']):
            self.vision_blocks.append(VisionBlock(options, block_index=i))

        self.define_derivatives()
        self.worker = worker

    def update_teacher_features_blocks(self):
        for v in self.vision_blocks:
            v.update_teacher_features_block()

    def define_derivatives(self):
        self.sobel_dx_kernel = torch.Tensor([[1 / 2, 0, -1 / 2],
                                             [1, 0, -1],
                                             [1 / 2, 0, -1 / 2]]).to(self.device)
        self.sobel_dy_kernel = torch.Tensor([[1 / 2, 1, 1 / 2],
                                             [0, 0, 0],
                                             [-1 / 2, -1, -1 / 2]]).to(self.device)

        self.hs_dx_kernel = torch.Tensor([[0, 0, 0],
                                          [0, -1 / 4, 1 / 4],
                                          [0, -1 / 4, 1 / 4]]).to(self.device)

        self.hs_dy_kernel = torch.Tensor([[0, 0, 0],
                                          [0, -1 / 4, -1 / 4],
                                          [0, 1 / 4, 1 / 4]]).to(self.device)

        self.hs_dt_kernel = torch.Tensor([[0, 0, 0],
                                          [0, 1 / 4, 1 / 4],
                                          [0, 1 / 4, 1 / 4]]).to(self.device)

        self.hs_filter = torch.Tensor([[1 / 12, 1 / 6, 1 / 12],
                                       [1 / 6, 0, 1 / 6],
                                       [1 / 12, 1 / 6, 1 / 12]]).to(self.device)

        if self.options['c'] == 3:
            self.hs_dt_kernel_f = self.hs_dt_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.hs_dx_kernel_f = self.hs_dx_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.hs_dy_kernel_f = self.hs_dy_kernel.view((1, 1, 3, 3)).expand(3, -1, -1, -1)
            self.sobel_dx_kernel_f = self.sobel_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
            self.sobel_dy_kernel_f = self.sobel_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        elif self.options['c'] == 1:
            self.hs_dt_kernel_f = self.hs_dt_kernel.view((1, 1, 3, 3))
            self.hs_dx_kernel_f = self.hs_dx_kernel.view((1, 1, 3, 3))
            self.hs_dy_kernel_f = self.hs_dy_kernel.view((1, 1, 3, 3))
            self.sobel_dx_kernel_f = self.sobel_dx_kernel.view((1, 1, 3, 3))
            self.sobel_dy_kernel_f = self.sobel_dy_kernel.view((1, 1, 3, 3))
        else:
            raise NotImplementedError

        self.hs_dx_kernel_uv = self.hs_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.hs_dy_kernel_uv = self.hs_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.sobel_filter_x_uv = self.sobel_dx_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)
        self.sobel_filter_y_uv = self.sobel_dy_kernel.view((1, 1, 3, 3)).expand(2, -1, -1, -1)

    def displacement_gradient(self, x, x_=None, type="sobel"):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        batch_dims = x.shape[0]
        channel_dims = x.shape[1]

        if type == "shift":
            left = x
            right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
            top = x
            bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

            dx, dy = right - left, bottom - top
            # dx will always have zeros in the last column, right-left
            # dy will always have zeros in the last row,    bottom-top
            dx[:, :, :, -1] = 0
            dy[:, :, -1, :] = 0

        elif type == "sobel":
            dx, dy = self.sobel_gradient(x)

        elif type == "hs":
            assert channel_dims == 2  # on flow
            a = self.hs_dx_kernel_uv
            b = self.hs_dy_kernel_uv
            dx = F.conv2d(x, a, stride=1, padding=1, groups=channel_dims) * 2
            dy = F.conv2d(x, b, stride=1, padding=1, groups=channel_dims) * 2
        else:
            raise NotImplementedError
        return dx, dy

    def sobel_gradient(self, x):
        channels = x.shape[1]
        grad_x_weights = self.sobel_dx_kernel
        grad_y_weights = self.sobel_dy_kernel

        grad_x_weights = grad_x_weights.expand(channels, 1, 3, 3)
        grad_y_weights = grad_y_weights.expand(channels, 1, 3, 3)

        padded_x = F.pad(x, (1, 1, 1, 1), "replicate")
        grad_x = F.conv2d(padded_x, grad_x_weights, groups=channels)
        grad_y = F.conv2d(padded_x, grad_y_weights, groups=channels)
        return grad_x, grad_y

    def feature_gradient(self, features):
        return self.sobel_gradient(features)

    def forward(self, current_frame, old_frame, motion_disk=None):
        features_current = []
        features_old = []
        logits_current = []
        logits_old = []
        lower_features_current = []
        lower_features_old = []
        displacements_fwd = []
        displacements_bwd = []
        inputs = current_frame, old_frame

        for i in range(self.options['n_blocks']):  # current_frame, old_frame, coarser_displacements=None, initial_input=None

            block_features, lower_block_features, block_displacements, block_logits = \
                self.vision_blocks[i](*inputs, motion_disk=motion_disk)

            features_old.append(block_features[1])
            features_current.append(block_features[0])
            ###
            logits_old.append(block_logits[1])
            logits_current.append(block_logits[0])
            ###
            lower_features_old.append(lower_block_features[1])
            lower_features_current.append(lower_block_features[0])
            ###
            displacements_fwd.append(block_displacements['fwd'])
            if 'bwd' in block_displacements:
                displacements_bwd.append(block_displacements['bwd'])
            inputs = block_features
        if self.options['vision_block']['displacements']['of_only']:
            for i in range(1, self.options['n_blocks']):
                displacements_fwd[i] = displacements_fwd[0]
        displacements = {'fwd': displacements_fwd, 'bwd': displacements_bwd}
        return displacements, features_current, features_old, lower_features_current, lower_features_old, \
               logits_current, logits_old

    def compute_frame_prediction(self, current_features, displacements):
        if self.options['decoder_input'] == 'concatenated':
            disp = displacements["fwd"][-1]
            prediction_input = [torch.cat(current_features, 1)], [disp]
        elif self.options['decoder_input'] == 'last-block':
            # prediction_input = [current_features[-1]], displacements # TODO bug!
            disp = displacements["fwd"][-1]
            prediction_input = [current_features[-1]], [disp]
        elif self.options['decoder_input'] == 'per-block':  # TODO case of scratch_displacement is not supported
            prediction_input = current_features, displacements["fwd"]
        return self.decoder(*prediction_input)

    def compute_masked_frame_prediction(self, current_features, displacements):
        if self.options['decoder_input'] == 'concatenated':
            disp = displacements["fwd"][-1]
            prediction_input = [torch.cat(current_features, 1)], [disp]
        elif self.options['decoder_input'] == 'last-block':
            # prediction_input = [current_features[-1]], displacements # TODO bug!
            disp = displacements["fwd"][-1]
            prediction_input = [current_features[-1]], [disp]
        elif self.options['decoder_input'] == 'per-block':  # TODO case of scratch_displacement is not supported
            prediction_input = current_features, displacements["fwd"]
        else:
            raise NotImplementedError

        mode = self.options['masked_decoder_mode']

        feat, displ = prediction_input
        # cycle to get average feature in the mask
        binary_flag = "bsquash"
        g_m = self.options['vision_block']['gamma_m']
        output_averaged_tensor = []

        for block in range(len(feat)):
            m = torch.sum((displ[block].detach() ** 2), dim=1, keepdim=True)
            mask = squash(m, gamma=g_m, binary=binary_flag) == True
            # masked_features = torch.masked_select(feat[block], mask) # NOTICE: in this way, all channels are mixed up - ALSO THE BATCH!
            # mean_value = torch.mean(masked_features)  # unique value or channel-wise? TODO deve essere batch-wise!
            # # inject this value in the returned tensor
            # new_features = torch.zeros_like(feat[block]).to(feat[block].device)
            # filled_tensor = torch.masked_fill(new_features, mask, mean_value)
            # output_averaged_tensor.append(filled_tensor)
            b, f, h, w = feat[block].shape

            new_features = torch.zeros_like(feat[block]).to(feat[block].device)

            if mode == "channel_avg":
                masked_feat = feat[block] * mask  # [b, f, h, w]
                active_locations = torch.sum(mask, dim=[2, 3])  # [b, 1]
                channel_avg = torch.sum(masked_feat, dim=[2, 3]) / (active_locations + self.eps)  # [b, f]
                broadcast_avg = channel_avg[..., None, None].expand(b, f, h, w)  # [b, f, 1, 1] => [b, f, h, w]
                # NOTICE: must have same size of new_features

                # new_features = torch.masked_scatter(new_features, mask, broadcast_avg)
                # https://stackoverflow.com/questions/68675160/torch-masked-scatter-result-did-not-meet-expectations
                # best thing: broadcast_avg has the needed format; select the index that are true in the mask and do indexing!
                # new_features.masked_scatter_(mask, channel_avg)  # TODO check this code!

            elif mode == "global_avg":
                masked_feat = feat[block] * mask  # [b, f, h, w]
                active_locations = torch.sum(mask, dim=[2, 3]).squeeze()
                global_avg = torch.sum(masked_feat, dim=[1, 2, 3]) / (active_locations + self.eps)
                broadcast_avg = global_avg[:, None, None, None].expand(b, f, h, w)
                # new_features.masked_scatter_(mask, global_avg)
                # new_features = torch.masked_scatter(new_features, mask, broadcast_avg)

            else:
                raise NotImplementedError
            # ugly code for for the batch, otherwise when indexing it looses the batch dim
            # batch_size = feat[block].shape[0]
            # for batch in range(batch_size):
            #     example_feat, example_displ, example_mask = feat[block][batch], displ[block][batch], mask[batch]
            #     masked_features = torch.masked_select(example_feat,
            #                                           example_mask)  # in this way, channel-wise info is lost!
            #     mean_value = torch.mean(masked_features)  # maybe better to have the mean over each channel?
            #     # inject this value in the returned tensor
            #     new_features[batch].masked_fill_(example_mask, mean_value)  # in-place

            # flattened_mask = mask.expand(b, f, h, w).flatten()
            # new_features = new_features.flatten()
            # broadcast_avg = broadcast_avg.flatten()
            # new_features[flattened_mask] = broadcast_avg[flattened_mask]
            # new_features = new_features.view(b, f, h, w)
            expanded_mask = mask.expand(b, f, h, w)
            new_features[expanded_mask] = broadcast_avg[expanded_mask]

        output_averaged_tensor.append(new_features)
        # import lve
        # lve.utils.plot_grid(new_features[0].detach().cpu().numpy())  # for debugging!
        # lve.utils.plot_grid(broadcast_avg[0].detach().cpu().numpy())

        prediction_input = output_averaged_tensor, displacements["fwd"]
        return self.decoder(*prediction_input)

    # def compute_lists_gradients(self, old_features, features, lower_features_old, lower_features_current,
    #                             displacements):
    #     d_old_feat, d_feat, d_old_feat_lower, d_feat_lower, d_flow = [], [], [], [], []
    #     for i in range(self.options['n_blocks']):
    #         old_feat_dx, old_feat_dy = self.feature_gradient(old_features[i])  # [b, c, h, w ]
    #         d_old_feat.append([old_feat_dx, old_feat_dy])
    #         feat_dx, feat_dy = self.feature_gradient(features[i])  # [b, c, h, w ]
    #         d_feat.append([feat_dx, feat_dy])
    #         flow_dx, flow_dy = self.gradient(displacements[i], type=self.options['gradient_type'])  # [b, 2, h, w]
    #         d_flow.append([flow_dx, flow_dy])
    #     return d_old_feat, d_feat, d_flow

    def compute_variance_loss(self, type):
        variance_loss = []
        for k, v in self.named_parameters():
            if "weight" in k and "features_block" in k and v.ndim == 4:
                if type == 'intra':
                    variance_loss.append(v.var(dim=[1, 2, 3]).mean())  # [out, in , k1, k2]
                elif type == 'inter':
                    # variance among filters
                    variance_loss.append(v.var(dim=[0]).mean())
        return - torch.mean(torch.stack(variance_loss))

    def compute_gradients_list(self, tensor, type="features"):
        list_grad = []
        for i in range(self.options['n_blocks']):
            if type == "features":
                tensor_dx, tensor_dy = self.feature_gradient(tensor[i])
            elif type == "displacements":
                tensor_dx, tensor_dy = self.displacement_gradient(tensor[i], type=self.options['gradient_type'])
            else:
                raise NotImplementedError
            list_grad.append([tensor_dx, tensor_dy])
        return list_grad

    def compute_loss(self, features, old_features, lower_features_current, lower_features_old,
                     displacements, frame, old_frame, predicted_frame):
        # old_features = [old_feat.detach() for old_feat in old_features]  # WARNING: detaching old features here!

        # used whenever a loss is not computed (see following)
        zero_list = [torch.zeros(1).to(frame.device) for i in range(self.options["n_blocks"])]
        batched_zero_list = [torch.zeros(frame.shape[0], 1).to(frame.device) for i in range(self.options["n_blocks"])]
        empty_list = [[] for i in range(self.options["n_blocks"])]

        if self.options['teacher'] or self.options['detach_cur']:
            features = [feat.detach() for feat in features]

        # compute here gradients, only once
        # d_old_feat = self.compute_gradients_list(old_features)
        d_feat = None

        d_feat_lower = None
        d_flow = self.compute_gradients_list(displacements['fwd'], type="displacements")

        # list with detached displacements
        detached_displacements = {'fwd': [displ.clone().detach() for displ in displacements['fwd']],
                                  'bwd': [displ.clone().detach() for displ in displacements['bwd']]}

        # original consistency loss
        if sum(self.options["lambda_c_lower"]) > 0.:  # do not compute if not requested => speed-up
            detached_lower_features_current = [feat.detach() for feat in lower_features_current]
            detached_lower_features_old = [feat.detach() for feat in lower_features_old]
            consistency_loss_lower, consistency_2d_list_lower, consistency_list_lower, occl_list_lower = self.compute_consistency_loss(
                detached_lower_features_current,
                detached_lower_features_old,
                displacements,
                lambdas=self.options["lambda_c_lower"],
                loss_type=self.options["loss_type"],
                occ=self.options["vision_block"]["displacements"]["occlusion"],
                level="lower")

            # cross-layer consistency loss

        else:
            consistency_loss_lower, consistency_2d_list_lower, consistency_list_lower, occl_list_lower = torch.zeros(
                1).to(
                frame.device), None, zero_list, None

        if sum(self.options["lambda_c_upper"]) > 0.:
            consistency_loss_upper, consistency_2d_list_upper, consistency_list_upper, occl_list_upper = self.compute_consistency_loss(
                features,
                old_features,
                detached_displacements,
                lambdas=
                self.options[
                    "lambda_c_upper"],
                loss_type=
                self.options[
                    "loss_type"],
                level="upper",
                occ=
                self.options[
                    "vision_block"][
                    "displacements"][
                    "occlusion"])
            consistency_metric_dict = self.compute_consistency_metrics(features, old_features, displacements['fwd'])
        else:
            consistency_loss_upper, consistency_2d_list_upper, consistency_list_upper, occl_list_upper = torch.zeros(
                1).to(
                frame.device), None, zero_list, None
            consistency_metric_dict = {"consistency_absolute_metric": torch.zeros(1).to(frame.device),
                                       "consistency_relative_metric": torch.zeros(1).to(frame.device),
                                       "consistency_absolute_list": zero_list,
                                       "consistency_absolute_2d_list": batched_zero_list,
                                       "consistency_relative_list": zero_list,
                                       "consistency_relative_2d_list": batched_zero_list,
                                       }

        if sum(self.options["lambda_c_skip"]) > 0.:  # do not compute if not requested => speed-up
            disp = detached_displacements if self.options["vision_block"]["displacements"]['consistency_skiploss'] == 'features' else displacements
            if self.options['vision_block']['displacements']['consistency_skiploss'] == 'features':
                cur_feat = features
                old_feat = old_features
            elif self.options['vision_block']['displacements']['consistency_skiploss'] == 'motion':
                cur_feat = lower_features_current
                old_feat = lower_features_old
            else:
                raise NotImplementedError

            consistency_loss_skip, consistency_2d_list_skip, consistency_list_skip, occl_list_skip = self.compute_consistency_loss(
                cur_feat,
                old_feat,
                disp,
                lambdas=self.options["lambda_c_skip"],
                loss_type='charb',
                occ=self.options["vision_block"]["displacements"]["occlusion"],
                level="skip")
        else:
            consistency_loss_skip, consistency_2d_list_skip, consistency_list_skip, occl_list_skip = torch.zeros(
                1).to(frame.device), None, zero_list, None

        # motion edge-aware smoothness - lambda_s
        # do not compute if not requested => speed-up
        if sum(self.options['vision_block']['displacements']['lambda_s']) > 0.:
            # detach is done inside!
            # notice: edge-aware wrt features incoming to the block (RGB in the first block)
            smoothness_dic, edge_aware_mmask_list_x, edge_aware_mmask_list_y \
                = self.compute_motion_smoothness_features_edge_aware_loss(d_feat_lower, d_flow)
            smoothness_loss = smoothness_dic['motion_smoothness']
            smoothness_w_loss = smoothness_dic['motion_smoothness_w']
        else:
            smoothness_loss, smoothness_w_loss = torch.zeros(1).to(frame.device), torch.zeros(1).to(frame.device)

        # feature spatial coherence - not implemented
        spatial_coherence_loss_sum, spatial_coherence_w_loss_sum = torch.zeros(1).to(frame.device), torch.zeros(
                1).to(frame.device)
        feature_spatial_coherence_dic = {"fsmoothness_2d": None, "super_features_2d":None, "out_features_2d":None, "m_2d": None, "cvx_2d": None, "cvy_2d": None, "dfx_2d": None, "dfy_2d": None}

        # not implemented
        spatial_coherence_tv_loss_sum, spatial_coherence_tv_w_loss_sum = torch.zeros(1).to(frame.device), torch.zeros(1).to(frame.device)

        if self.options['lambda_r'] > 0.:
            regularization_loss = compute_regularization_loss(displacements['fwd'])
        else:
            regularization_loss = torch.zeros(1).to(frame.device)


        if self.options['lambda_sim'] > 0.:
            input_features1 = features if self.options["vision_block"]["displacements"]["motion_disk_type"] == "motion_unity" else old_features
            input_features2 = old_features if self.options["vision_block"]["displacements"][
                                              "motion_disk_type"] == "motion_unity" else features
            similarity_loss, _, similarity_list, dissimilarity_list, simdissim_idxs_list = \
                self.compute_similarity_dissimilarity_loss(input_features1, input_features2, displacements['fwd'])
        else:
            similarity_loss, dissimilarity_loss, similarity_list, dissimilarity_list, simdissim_idxs_list = \
                torch.zeros(1).to(frame.device), torch.zeros(1).to(frame.device), zero_list, zero_list, empty_list

        total_loss = consistency_loss_lower + \
                     smoothness_w_loss + spatial_coherence_w_loss_sum + spatial_coherence_tv_w_loss_sum + \
                      + self.options['lambda_r'] * regularization_loss + \
                      consistency_loss_upper + consistency_loss_skip + similarity_loss

        dic = {
            'consistency_lower': consistency_loss_lower,
            'consistency_upper': consistency_loss_upper,
            'consistency_upper_2d': consistency_2d_list_upper,
            'spatial_coherence': spatial_coherence_loss_sum, 'spatial_coherence_w': spatial_coherence_w_loss_sum,
            'feature_total_variation': spatial_coherence_tv_loss_sum,
            'feature_total_variation_w': spatial_coherence_tv_w_loss_sum,
            'motion_smoothness': smoothness_loss, 'motion_smoothness_w': smoothness_w_loss,
            'similarity_loss': similarity_loss,
            'simdissim_points': simdissim_idxs_list,
            'total': total_loss,
            'fsmoothness_2d': feature_spatial_coherence_dic['fsmoothness_2d'],
            'occl_2d_upper': occl_list_upper,
            'occl_2d_lower': occl_list_lower,
            'd_feat_x_2d': [torch.abs(d_f[0]) for d_f in d_feat] if d_feat is not None else empty_list,
            'd_feat_y_2d': [torch.abs(d_f[1]) for d_f in d_feat] if d_feat is not None else empty_list,
            'd_flow_x_2d': [torch.abs(d_f[0]) for d_f in d_flow],
            'd_flow_y_2d': [torch.abs(d_f[1]) for d_f in d_flow],
            'consistency_absolute': consistency_metric_dict["consistency_absolute_metric"],
            'consistency_relative': consistency_metric_dict["consistency_relative_metric"],
            'consistency_relative_2d': consistency_metric_dict["consistency_relative_2d_list"],
            'consistency_absolute_2d': consistency_metric_dict["consistency_absolute_2d_list"]}
        dic.update({'dissimilarity_b' + str(i): dissimilarity_list[i] for i in range(len(dissimilarity_list))})
        dic.update({'similarity_b' + str(i): similarity_list[i] for i in range(len(similarity_list))})
        dic.update(
            {'consistency_lower_b' + str(i): consistency_list_lower[i] for i in range(len(consistency_list_lower))})
        dic.update(
            {'consistency_upper_b' + str(i): consistency_list_upper[i] for i in range(len(consistency_list_upper))})
        dic.update(
            {'consistency_skip_b' + str(i): consistency_list_skip[i] for i in range(len(consistency_list_skip))})
        dic.update({'consistency_absolute_b' + str(i): consistency_metric_dict["consistency_absolute_list"][i] for i in
                    range(len(consistency_metric_dict["consistency_absolute_list"]))})
        dic.update({'consistency_relative_b' + str(i): consistency_metric_dict["consistency_relative_list"][i] for i in
                    range(len(consistency_metric_dict["consistency_relative_list"]))})

        return dic


    def compute_smoothness_loss(self, displacements):
        smoothness_terms = []
        smoothness_w_terms = []
        for i in range(self.options['n_blocks']):
            flow_dx, flow_dy = self.gradient(displacements[i], type=self.options['gradient_type'])
            flow_norm = 0.5 * torch.mean(flow_dx ** 2 + flow_dy ** 2)
            smoothness_terms.append(flow_norm)
            smoothness_w_terms.append(flow_norm *
                                      self.options['vision_block']['displacements']['lambda_s'][i])

        return {'smoothness': torch.mean(torch.stack(smoothness_terms)),
                'smoothness_w': torch.mean(torch.stack(smoothness_w_terms))}

    def compute_motion_smoothness_features_edge_aware_loss(self, d_feat, d_flow):  # still to be used
        smoothness_terms = []
        smoothness_w_terms = []
        edge_aware_mask_list_x = []
        edge_aware_mask_list_y = []

        for i in range(self.options['n_blocks']):
            if d_feat is None:
                exp_x = 1.0
                exp_y = 1.0

            flow_dx, flow_dy = d_flow[i]
            flow_dx = flow_dx ** 2 * exp_x
            flow_dy = flow_dy ** 2 * exp_y
            flow_norm = torch.mean(flow_dx + flow_dy)
            smoothness_terms.append(flow_norm)
            smoothness_w_terms.append(flow_norm *
                                      self.options['vision_block']['displacements']['lambda_s'][i])

        return {'motion_smoothness': torch.mean(torch.stack(smoothness_terms)),
                'motion_smoothness_w': torch.mean(
                    torch.stack(smoothness_w_terms))}, edge_aware_mask_list_x, edge_aware_mask_list_y

    def compute_consistency_loss(self, features, old_features, displacements, lambdas, loss_type="charb",
                                 level="lower", occ=None):

        occl_list = []
        consistency_list = []
        consistency_2d_list = []
        consistency_list_w = []

        for b in range(len(features)):  # to support also cross layer consistency
            if (b == 0 and level == "lower") or level == 'skip':
                # for the RGB values we do not want to use xent, but the standard charb
                loss_type_local = "charb"
            else:
                loss_type_local = loss_type

            if self.options["consistency_type"] == "masked" and level == "upper":
                moving_threshold = self.options['moving_threshold']
                motion_norm = torch.linalg.norm(displacements['fwd'][b], dim=1, keepdim=True)
                norm_average = torch.mean(motion_norm, dim=[2, 3])  # b x 1
                moving_threshold = torch.clamp(norm_average, min=moving_threshold)  # max(norm, thresh) - b x 1
                moving = motion_norm >= moving_threshold.view(-1, 1, 1, 1)
                moving_mask = moving.to(torch.float)
            else:
                moving_mask = None

            if occ == 'back':
                occl_mask = 1 - get_occu_mask_backward(displacements['bwd'][b], th=0.2)
            elif occ == 'bidir':
                occl_mask = 1 - get_occu_mask_bidirection(displacements['fwd'][b], displacements['bwd'][b])
            else:
                occl_mask = None

            if level == "skip":
                if b == 0:
                    loss_b, loss_2d = torch.tensor(0).to(features[0].device), None

                elif self.options['vision_block']['displacements']['consistency_skiploss'] == 'features':
                    loss_b, loss_2d = compute_photometric_loss(old_features[b], backward_warp(
                                                           frame=features[b],
                                                           displacement=displacements['fwd'][0]),
                                                           charb_eps=self.options['charb_eps'],
                                                           charb_alpha=self.options['charb_alpha'],
                                                           loss_type=loss_type_local,
                                                           occlusion_mask=occl_mask, moving_mask=moving_mask)
                elif self.options['vision_block']['displacements']['consistency_skiploss'] == 'motion':
                    loss_b, loss_2d = compute_photometric_loss(old_features[0], backward_warp(
                                                            frame=features[0],
                                                            displacement=displacements['fwd'][b]),
                                                            charb_eps=self.options['charb_eps'],
                                                            charb_alpha=self.options['charb_alpha'],
                                                            loss_type=loss_type_local,
                                                            occlusion_mask=occl_mask, moving_mask=moving_mask)
            else:
                loss_b, loss_2d = compute_photometric_loss(old_features[b],
                                                           backward_warp(frame=features[b],
                                                            displacement=displacements['fwd'][b]),
                                                       charb_eps=self.options['charb_eps'],
                                                       charb_alpha=self.options['charb_alpha'],
                                                       loss_type=loss_type_local,
                                                       occlusion_mask=occl_mask, moving_mask=moving_mask)

            # loss_b = torch.log(loss_b)
            consistency_list.append(loss_b)
            consistency_2d_list.append(loss_2d)
            consistency_list_w.append(lambdas[b] * loss_b)
            occl_list.append(occl_mask)

        consistency_loss = torch.mean(torch.stack(consistency_list_w))
        return consistency_loss, consistency_2d_list, consistency_list, occl_list

    def compute_consistency_metrics(self, features, old_features, displacements):

        consistency_absolute_list = []
        consistency_relative_list = []
        consistency_relative_2d_list = []
        consistency_absolute_2d_list = []
        with torch.no_grad():
            for old_feat, feat, displacement in zip(old_features, features, displacements):
                f1 = old_feat
                f2 = backward_warp(frame=feat, displacement=displacement)
                diff = f1 - f2

                moving_threshold = self.options['moving_threshold']
                moving = torch.linalg.norm(displacement, dim=1, keepdim=True) >= moving_threshold
                moving_mask = moving.to(torch.float)

                diff = diff * moving_mask
                diff_scaled = 2 * diff / (f1 + f2 + self.eps)
                consistency_absolute_list.append(torch.abs(diff).mean())
                consistency_relative_list.append(torch.abs(diff_scaled).mean())
                consistency_relative_2d_list.append(diff_scaled)
                consistency_absolute_2d_list.append(diff)

            consistency_absolute_metric = torch.mean(torch.stack(consistency_absolute_list))
            consistency_relative_metric = torch.mean(torch.stack(consistency_relative_list))

        return {"consistency_absolute_metric": consistency_absolute_metric,
                "consistency_absolute_list": consistency_absolute_list,
                "consistency_absolute_2d_list": consistency_absolute_2d_list,
                "consistency_relative_metric": consistency_relative_metric,
                "consistency_relative_list": consistency_relative_list,
                "consistency_relative_2d_list": consistency_relative_2d_list,
                }

    def compute_dist_ground_truth(self, flows, motions, motion_mask, predicted_motion_mask, eps_error=1.0):
        intersection_moving_mask = torch.logical_and(motion_mask, predicted_motion_mask)
        still_mask = torch.logical_not(motion_mask)
        cos = torch.nn.CosineSimilarity(dim=1)
        angular_errors = torch.acos(torch.clamp(cos(flows, motions), min=-1., max=1.)).unsqueeze(dim=1)

        angular_error = torch.mean(angular_errors) * 180. / math.pi
        # angular_error_moving = torch.mean(angular_errors[moving_mask]) * 180. / math.pi
        angular_error_moving = torch.mean(angular_errors[intersection_moving_mask]) * 180. / math.pi
        angular_error_still = torch.mean(angular_errors[still_mask]) * 180. / math.pi

        if torch.isnan(angular_error_moving): angular_error_moving = torch.tensor(0.0)
        if torch.isnan(angular_error_still): angular_error_still = torch.tensor(0.0)

        # euclid_dist = torch.sqrt(torch.sum((flows - motions) ** 2, dim=1))
        # l2_dist = torch.mean(euclid_dist)
        # l2_dist_moving = torch.mean(euclid_dist[moving_mask])
        # l2_dist_still = torch.mean(euclid_dist[still_mask])
        # error_rate = torch.mean((euclid_dist > eps_error).float())
        # if torch.isnan(l2_dist_moving): l2_dist_moving = torch.tensor(0.0)
        # if torch.isnan(l2_dist_still): l2_dist_still = torch.tensor(0.0)

        return {
            # 'l2_dist': l2_dist, 'l2_dist_moving': l2_dist_moving, 'l2_dist_still': l2_dist_still,
            # 'error_rate': error_rate,
            'angular_error': angular_error, 'angular_error_moving': angular_error_moving,
            'angular_error_still': angular_error_still
        }

    def compute_similarity_dissimilarity_loss(self, features1, features2, flow):
        d_sim_list = []
        d_sim_list_w = []
        d_dis_list = []
        idxs_list = []

        for b in range(len(features1)):
            d_sim_loss, d_dis_loss, idxs = sampled_similarity_dissimilarity_loss(
                features1[b], features2[b], flow[b],
                sampled_pairs=self.options['num_pairs'],
                dissimilarity_threshold=self.options['dissimilarity_threshold'],
                similarity_threshold=self.options['similarity_threshold'],
                moving_threshold=self.options['moving_threshold'],
                moving_vs_static_only=self.options['moving_vs_static_only'],
                simdis_type=self.options['simdis_type'],
                simdis_loss_tau=self.options['simdis_loss_tau'],
                simdis_loss=self.options['simdis_loss'],
                sampling_type=self.options['sampling_type'],
                kept_pairs_perc=self.options['kept_pairs_perc'],
                sampling_features=self.options['sampling_features'],
                simdis_neg_avg=self.options['simdis_neg_avg'],
            )

            d_sim_list.append(d_sim_loss)
            d_dis_list.append(d_dis_loss)
            d_sim_list_w.append(d_sim_loss * self.options['lambda_sim'])
            idxs_list.append(idxs)

        d_sim_loss_w = torch.mean(torch.stack(d_sim_list_w))
        return d_sim_loss_w, None, d_sim_list, d_dis_list, idxs_list

    def compute_similarity_loss(self, features, flow):
        d_list = []
        d_list_w = []
        mask_list = []

        for b in range(len(features)):
            d_loss, mask = sampled_similarity_loss(features[b], flow[b],
                                                   sampled_pairs=self.options['num_pairs'],
                                                   similarity_threshold=0.0,
                                                   moving_threshold=0.1,
                                                   all_that_move=False)
            d_list.append(d_loss)
            d_list_w.append(d_loss * self.options['lambda_sim'])
            mask_list.append(mask)

        d_loss_w = torch.mean(torch.stack(d_list_w))
        return d_loss_w, d_list, mask_list

    def zero_grad(self):
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()

    def compute_weights_norm(self):
        w = 0.0
        b = 0.0
        for param in self.parameters():
            if param.ndim == 1:
                b += torch.sum(param ** 2)
            else:
                w += torch.sum(param ** 2)
        if torch.is_tensor(b): b = b.item()
        if torch.is_tensor(w): w = w.item()
        return w, b

    def print_parameters(self):
        params = list(self.parameters())
        print("Number of tensor params: " + str(len(params)))
        for i in range(0, len(params)):
            p = params[i]
            print("   Tensor size: " + str(p.size()) + " (req. grad = " + str(p.requires_grad) + ")")
