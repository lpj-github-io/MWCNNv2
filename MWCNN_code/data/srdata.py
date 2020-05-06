import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio
from scipy.misc import imresize

import torch
import torch.utils.data as data
import h5py

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        if train:
            mat = h5py.File('../MWCNN/imdb_gray.mat')
            self.args.ext = 'mat'
            self.hr_data = mat['images']['labels'][:,:,:,:]
            self.num = self.hr_data.shape[0]
            print(self.hr_data.shape)

        if self.split == 'test':
            self._set_filesystem(args.dir_data)

        self.images_hr = self._scan()



    def _scan(self):
        raise NotImplementedError
    #
    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    # def _name_hrbin(self):
    #     raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        if self.train:


            lr, hr, scale = self._get_patch(hr, filename)

            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
            return lr_tensor, hr_tensor, filename
        else:
            #scale = 2
            # scale = self.scale[self.idx_scale]
            lr, hr, _ = self._get_patch(hr, filename)

            lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)

            return lr_tensor, hr_tensor, filename


    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        # lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]

        if self.args.ext == 'img' or self.benchmark:
            filename = hr

            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            # lr = np.load(lr)
            hr = np.load(hr)
        elif self.args.ext == 'mat' or self.train:
            hr = self.hr_data[idx, :, :, :]
            hr = np.squeeze(hr.transpose((1, 2, 0)))
            filename = str(idx) + '.png'
        else:
            filename = str(idx + 1)




        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return hr, filename

    def _get_patch(self, hr, filename):
        patch_size = self.args.patch_size

        if self.train:
            scale = self.scale[0]
            if self.args.task_type == 'denoising':
                lr, hr = common.get_patch_noise(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'SISR':
                lr, hr = common.get_patch_bic(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'JIAR':
                lr, hr = common.get_patch_compress(
                    hr, patch_size, scale
                )

            lr, hr = common.augment([lr, hr])
            return lr, hr, scale
        else:
            scale = self.scale[0]
            if self.args.task_type == 'denoising':
                lr, hr = common.add_img_noise(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'SISR':
                lr, hr = self._get_patch_test(
                    hr, patch_size, scale
                )
            if self.args.task_type == 'JIAR':
                lr, hr = common.get_img_compress(
                    hr, patch_size, scale
                )
            return lr, hr, scale
            # lr = common.add_noise(lr, self.args.noise)


    def _get_patch_test(self, hr, scale):

        ih, iw = hr.shape[0:2]
        lr = imresize(imresize(hr, [int(ih/scale), int(iw/scale)], 'bicubic'), [ih, iw], 'bicubic')
        ih = ih // 8 * 8
        iw = iw // 8 * 8
        hr = hr[0:ih, 0:iw, :]
        lr = lr[0:ih, 0:iw, :]

        return lr, hr




    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

