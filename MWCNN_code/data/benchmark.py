import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):


        if self.train:
            list_hr = [i for i in range(self.num)]
        else:
            list_hr = []

            for entry in os.scandir(self.dir_hr):
                filename = os.path.splitext(entry.name)[0]
                list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            list_hr.sort()

        return list_hr#, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.args.dir_data,self.args.data_test )
        self.ext = '*.*'
