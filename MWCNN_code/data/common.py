import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

import os
import torch.nn as nn
import math
import time

from scipy.misc import imread, imresize, imsave, toimage
from scipy.ndimage import convolve
from PIL import Image


def get_patch_noise(img_tar, patch_size, noise_level):

    ih, iw = img_tar.shape[0:2]
    a = np.random.rand(1)[0] * 0.2 + 0.8
    if (ih * a < patch_size) | (a * iw < patch_size):
        a = np.random.rand(1)[0] * 0.33 + 0.67
        # img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')
        # th, tw = img_tar.shape[:2]
    else:
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')

    th, tw = img_tar.shape[:2]
    tp = patch_size

    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)
    img_tar = np.expand_dims(img_tar, axis=2)
    #print(img_tar.shape)
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    tt, _ = math.modf(time.time())

    # tt = int(np.random.randint(0,1e13,1))
    if np.random.rand() > 0.75:
        out = "tmp/" + '%d' % (tt * 1e16) + '.jpg'
        x = Image.fromarray(np.squeeze(img_tar, axis=2))
        # x = toimage(x)
        # imsave(out, x, format='jpeg', quality=int(quality_factor))

        x.save(out, 'JPEG', quality=95)
        x = imread(out)
        os.remove(out)
        x = np.expand_dims(x, axis=2)
    else:
        x = img_tar

    noises = np.random.normal(scale=noise_level, size=x.shape)
    noises = noises.round()
    img_tar_noise = x.astype(np.int16) + noises.astype(np.int16)
    # x_noise = x_noise.clip(0, 255).astype(np.uint8)

    return img_tar_noise, img_tar


def add_img_noise(img_tar, noise_level):
    img_tar = np.expand_dims(img_tar, axis=2)
    #print(img_tar.shape)
    ih, iw = img_tar.shape[0:2]
    ih = int(ih//8*8)
    iw = int(iw//8*8)
    img_tar = img_tar[0:ih, 0:iw, :]
    noises = np.random.normal(scale=noise_level, size=img_tar.shape)
    noises = noises.round()
    img_tar_noise = img_tar.astype(np.int16) + noises.astype(np.int16)
    # x_noise = x_noise.clip(0, 255).astype(np.uint8)

    return img_tar_noise, img_tar


def get_patch_bic(img_tar, patch_size, scale_factor):

    ih, iw = img_tar.shape[0:2]
    a = np.random.rand(1)[0] * 0.8 + 0.2
    if (ih*a < patch_size) | (a*iw < patch_size):
        a = np.random.rand(1)[0] * 0.33 + 0.67
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')
        # th, tw = img_tar.shape[:2]
    else:
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')


    th, tw = img_tar.shape[:2]
    tp = patch_size

    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)
    img_lr = imresize(imresize(img_tar, [int(th/scale_factor), int(tw/scale_factor)], 'bicubic'), [th, tw], 'bicubic')
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    img_lr = img_lr[ty:ty + tp, tx:tx + tp, :]

    return img_lr, img_tar


def get_patch_compress(img_tar, patch_size, quality_factor):

    ih, iw = img_tar.shape[0:2]
    a = np.random.rand(1)[0] * 0.8 + 0.2
    if (ih*a < patch_size) | (a*iw < patch_size):
        a = np.random.rand(1)[0] * 0.33 + 0.67
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')
        # th, tw = img_tar.shape[:2]
    else:
        img_tar = imresize(img_tar, [int(ih * a), int(iw * a)], 'bicubic')


    th, tw = img_tar.shape[:2]
    tp = patch_size

    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)

    tt, _ = math.modf(time.time())
    out = "tmp/" + '%d' % (tt * 1e16) + '.jpg'
    img_lr = Image.fromarray(np.squeeze(img_tar, axis=2))
    img_lr.save(out, 'JPEG', quality=quality_factor)
    img_lr = imread(out)
    os.remove(out)
    img_lr = np.expand_dims(img_lr, axis=2)


    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    img_lr = img_lr[ty:ty + tp, tx:tx + tp, :]

    return img_lr, img_tar

def get_img_compress(img_tar, quality_factor):
    img_tar = np.expand_dims(img_tar, axis=2)
    ih, iw = img_tar.shape[0:2]
    ih = int(ih // 8 * 8)
    iw = int(iw // 8 * 8)
    img_tar = img_tar[0:ih, 0:iw, :]

    tt, _ = math.modf(time.time())
    out = "tmp/" + '%d' % (tt * 1e16) + '.jpg'
    img_lr = Image.fromarray(np.squeeze(img_tar, axis=2))
    img_lr.save(out, 'JPEG', quality=quality_factor)
    img_lr = imread(out)
    os.remove(out)
    img_lr = np.expand_dims(img_lr, axis=2)

    return img_lr, img_tar


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255.0)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]
