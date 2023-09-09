import glob
import os
from gzip import GzipFile

from torch import nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import kaiming_normal_, constant_

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from lve.utils import torch_float_01_to_np_uint8
import random


class SimpleMetric():
    def __init__(self, l):
        self.val = 0.0
        self.l = l

    def accumulate(self, v):
        self.val += v

    def get(self, d=None):
        return self.val / (self.l if d is None else d)

    def reset(self):
        self.val = 0.0


def parse_augmentation(args_cmd):
    colorjitter_prob = args_cmd.augmentation['colorjitter'][0]
    colorjitter_intensity = args_cmd.augmentation['colorjitter'][1]
    lower_bound = 1 - colorjitter_intensity
    upper_bound = 1 + colorjitter_intensity
    return {
        'brightness': (lower_bound, upper_bound),
        'contrast': (lower_bound, upper_bound),
        'hue': (- colorjitter_intensity, colorjitter_intensity),
        'saturation': (lower_bound, upper_bound),
        'prob': colorjitter_prob
    }


# def sample_jitter_params(brightness=(1.0, 1.0), contrast=(1.0, 1.0), saturation=(1.0, 1.0), hue=(0.0, 0.0)):
def sample_jitter_params(brightness=(0.7, 1.5), contrast=(0.8, 1.2), saturation=(0.4, 1.6), hue=(-0.3, 0.3), gaussian=(.1, 2)):
    b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
    c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
    s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
    h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))
    g = None if gaussian is None else float(torch.empty(1).uniform_(gaussian[0], gaussian[1]))
    return b, c, s, h, g


def jitter_frames(x, b, c, s, h, g):
    x = F.adjust_brightness(x, b)
    x = F.adjust_contrast(x, c)
    x = F.adjust_saturation(x, s)
    x = F.adjust_hue(x, h)
    x = F.gaussian_blur(x, 7, g)
    return x


class PairsOfFramesDataset(Dataset):

    def __init__(self, root_dir, device, force_gray=True, fix_motion_u=False, fix_motion_v=False, n=1,
                 input_normalization="no", motion_disk_type=None, augmentation=False, bgr_input=True):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(root_dir + os.sep + "frames" + os.sep + "**" + os.sep + "*.png", recursive=True))
        self.motion_disk_type = motion_disk_type
        if motion_disk_type is not None:
            self.motion_disk_type_files = sorted(
                glob.glob(root_dir + os.sep + motion_disk_type + os.sep + "**" + os.sep + "*.bin", recursive=True))

        self.motion_files = sorted(
            glob.glob(root_dir + os.sep + "motion" + os.sep + "**" + os.sep + "*.bin", recursive=True))
        self.motion_available = len(self.motion_files) > 0
        self.force_gray = force_gray
        self.length = len(self.files) - n  # remove last frame
        self.device = device
        self.fix_motion_u = fix_motion_u
        self.fix_motion_v = fix_motion_v
        self.n = n
        self.input_normalization = input_normalization == "yes"
        self.augmentation = augmentation
        self.bgr_input = bgr_input

    def __len__(self):
        return self.length

    def transform(self, old_frame, frame, motion, motion_flag=False):
        cropped_frame = frame
        cropped_old_frame = old_frame
        # Random resized crop
        if random.random() < self.augmentation['crop']:
            crop = transforms.RandomResizedCrop(size=old_frame.shape[1:])  # check if 256
            params = crop.get_params(old_frame, scale=(0.35, 1.0), ratio=(0.85, 1.1))

            cropped_old_frame = transforms.functional.resized_crop(old_frame, *params, size=old_frame.shape[1:])
            cropped_frame = transforms.functional.resized_crop(frame, *params, size=frame.shape[1:])
            if motion_flag:
                i, j, h, w = params
                h_ratio = h / old_frame.shape[1]  # ratio new_height/old_height
                w_ratio = w / old_frame.shape[2]
                motion = transforms.functional.resized_crop(motion, *params, size=motion.shape[1:])
                motion[0] = motion[0] / h_ratio
                motion[1] = motion[1] / w_ratio

        # cropped_old_frame = old_frame
        # cropped_frame = frame
        # Random horizontal flipping
        if self.augmentation['flip'] and random.random() < self.augmentation['flip']:
            cropped_old_frame = F.hflip(cropped_old_frame)
            cropped_frame = F.hflip(cropped_frame)
            if motion_flag:
                motion = F.hflip(motion)
                motion[0] = - motion[0]

        # Random vertical flipping
        if self.augmentation['flip'] and random.random() < self.augmentation['flip']:
            cropped_old_frame = F.vflip(cropped_old_frame)
            cropped_frame = F.vflip(cropped_frame)
            if motion_flag:
                motion = F.vflip(motion)
                motion[1] = - motion[1]

        if self.augmentation['colordropout'] > 0:
            dropout_mask = (torch.FloatTensor(3, 1, 1).uniform_().to(cropped_frame.device) > self.augmentation[
                'colordropout']).float()
            cropped_old_frame *= dropout_mask
            cropped_frame *= dropout_mask

        if self.augmentation['colorjitter']['prob'] > 0. and random.random() < self.augmentation['colorjitter']['prob']:
            b, c, s, h = sample_jitter_params(brightness=self.augmentation['colorjitter']['brightness'],
                                              contrast=self.augmentation['colorjitter']['contrast'],
                                              hue=self.augmentation['colorjitter']['hue'],
                                              saturation=self.augmentation['colorjitter']['saturation'])
            cropped_frame = jitter_frames(cropped_frame, b, c, s, h)
            cropped_old_frame = jitter_frames(cropped_old_frame, b, c, s, h)

        return cropped_old_frame, cropped_frame, motion

    def __getitem__(self, idx):
        old_frame = cv2.imread(self.files[idx])
        frame = cv2.imread(self.files[idx + self.n])
        if not self.bgr_input:
            # convert to rgb only ion case of False flag - notice that this is different from what done before
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)

        if self.force_gray and frame.shape[2] > 1:
            frame = np.reshape(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (frame.shape[0], frame.shape[1], 1))

        if self.motion_available:
            with GzipFile(self.motion_files[idx + self.n]) as f:
                motion = np.load(f)
                motion = torch.from_numpy(motion.transpose(2, 0, 1)).float()
            if self.fix_motion_v: motion[1] *= -1
            if self.fix_motion_u: motion[0] *= -1
        else:
            motion = torch.empty(1)

        if self.force_gray and old_frame.shape[2] > 1:
            old_frame = np.reshape(cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY),
                                   (old_frame.shape[0], old_frame.shape[1], 1))

        if self.motion_disk_type is not None:
            with GzipFile(self.motion_disk_type_files[idx]) as f:
                motion_disk_files = np.load(f)
                motion_disk_files = torch.from_numpy(motion_disk_files).float()
        else:
            motion_disk_files = torch.empty(1)

        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().div_(255.0)
        old_frame = torch.from_numpy(old_frame.transpose(2, 0, 1)).float().div_(255.0)
        if self.input_normalization:
            frame = (frame - 0.5) / 0.25
            old_frame = (old_frame - 0.5) / 0.25

        if self.augmentation:
            old_frame, frame, motion_disk_files = self.transform(old_frame, frame, motion_disk_files,
                                                                 self.motion_disk_type is not None)

        return (old_frame, frame, motion, motion_disk_files, idx)


def compute_farneback_motion(old_frame, frame, backward=False):
    if backward:
        frames = (frame, old_frame)
    else:
        frames = (old_frame, frame)
    return cv2.calcOpticalFlowFarneback(frames[0],
                                        frames[1],
                                        None,
                                        pyr_scale=0.4,
                                        levels=5,  # pyramid levels
                                        winsize=12,
                                        iterations=10,
                                        poly_n=5,
                                        poly_sigma=1.1,
                                        flags=0)


def compute_motions(frames, old_frames, backward=False):
    frames_numpy = torch_float_01_to_np_uint8(frames)
    old_frames_numpy = torch_float_01_to_np_uint8(old_frames)
    h, w = frames_numpy.shape[1:3]
    motions = []
    for i in range(frames_numpy.shape[0]):
        frame_gray = cv2.cvtColor(frames_numpy[i], cv2.COLOR_BGR2GRAY).reshape(h, w, 1)
        old_frame_gray = cv2.cvtColor(old_frames_numpy[i], cv2.COLOR_BGR2GRAY).reshape(h, w, 1)
        motions.append(compute_farneback_motion(frame=frame_gray, old_frame=old_frame_gray, backward=backward))
    return torch.tensor(np.array(motions)).permute(0, 3, 1, 2)

def recursive_avoid_bn(base_model):
    for id, (name, child_model) in enumerate(base_model.named_children()):
        if isinstance(child_model, nn.BatchNorm2d):
            setattr(base_model, name, nn.Identity())

    for name, immediate_child_module in base_model.named_children():
        recursive_avoid_bn(immediate_child_module)

def recursive_avoid_maxpool(base_model):
    for id, (name, child_model) in enumerate(base_model.named_children()):
        if isinstance(child_model, nn.MaxPool2d):
            setattr(base_model, name, nn.AvgPool2d(kernel_size=child_model.kernel_size,
                                                   stride=child_model.stride,
                                                   padding=child_model.padding,
                                                   ceil_mode=child_model.ceil_mode))

    for name, immediate_child_module in base_model.named_children():
        recursive_avoid_maxpool(immediate_child_module)

def fix_first_layer(x, c):
    orig_conv = x[0]
    x[0] = torch.nn.Conv2d(in_channels=c, out_channels=orig_conv.out_channels,
                                           kernel_size=orig_conv.kernel_size, stride=orig_conv.stride, groups=orig_conv.groups,
                                           dilation=orig_conv.dilation, padding_mode=orig_conv.padding_mode,
                                           padding=orig_conv.padding, bias=orig_conv.bias)
    return x


def recursive_reduce_num_filters(base_model, factor=2):
        for id, (name, child_model) in enumerate(base_model.named_children()):
            if isinstance(child_model, nn.Conv2d):
                orig_conv = child_model
                inplanes = int(orig_conv.in_channels // factor)
                if inplanes == 0: inplanes = 1
                new_conv = torch.nn.Conv2d(in_channels=inplanes,
                                                            out_channels=int(orig_conv.out_channels // factor),
                                                            kernel_size=orig_conv.kernel_size, stride=orig_conv.stride, groups=orig_conv.groups,
                                                            dilation=orig_conv.dilation, padding_mode=orig_conv.padding_mode,
                                                            padding=orig_conv.padding, bias=orig_conv.bias)
                setattr(base_model, name, new_conv)
            if isinstance(child_model, nn.BatchNorm2d):
                setattr(base_model, name, nn.BatchNorm2d(int(orig_conv.out_channels // factor)))

        for name, immediate_child_module in base_model.named_children():
            recursive_reduce_num_filters(immediate_child_module, factor)