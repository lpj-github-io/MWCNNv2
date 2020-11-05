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
        """
        Initialize dataset

        Args:
            self: (todo): write your description
            train: (todo): write your description
            benchmark: (todo): write your description
        """
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
        """
        Scan the given scan.

        Args:
            self: (todo): write your description
        """
        raise NotImplementedError
    #
    def _set_filesystem(self, dir_data):
        """
        Set the directory.

        Args:
            self: (todo): write your description
            dir_data: (str): write your description
        """
        raise NotImplementedError

    # def _name_hrbin(self):
    #     raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        """
        Return a tensor for a tensor.

        Args:
            self: (todo): write your description
            idx: (list): write your description
        """
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
        """
        Returns the length of the image.

        Args:
            self: (todo): write your description
        """
        return len(self.images_hr)

    def _get_index(self, idx):
        """
        Returns the index of the given index.

        Args:
            self: (todo): write your description
            idx: (str): write your description
        """
        return idx

    def _load_file(self, idx):
        """
        Loads an image from disk

        Args:
            self: (todo): write your description
            idx: (str): write your description
        """
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
        """
        Get an instance of a patch.

        Args:
            self: (todo): write your description
            hr: (str): write your description
            filename: (str): write your description
        """
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
        """
        Calculate test test function.

        Args:
            self: (todo): write your description
            hr: (str): write your description
            scale: (float): write your description
        """

        ih, iw = hr.shape[0:2]
        lr = imresize(imresize(hr, [int(ih/scale), int(iw/scale)], 'bicubic'), [ih, iw], 'bicubic')
        ih = ih // 8 * 8
        iw = iw // 8 * 8
        hr = hr[0:ih, 0:iw, :]
        lr = lr[0:ih, 0:iw, :]

        return lr, hr




    def set_scale(self, idx_scale):
        """
        Set scale.

        Args:
            self: (todo): write your description
            idx_scale: (str): write your description
        """
        self.idx_scale = idx_scale

