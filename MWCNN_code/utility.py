import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        """
        Initialize the internal state.

        Args:
            self: (todo): write your description
        """
        self.acc = 0
        self.tic()

    def tic(self):
        """
        Set the timer.

        Args:
            self: (todo): write your description
        """
        self.t0 = time.time()

    def toc(self):
        """
        Return the time as a list.

        Args:
            self: (todo): write your description
        """
        return time.time() - self.t0

    def hold(self):
        """
        Hold the next row

        Args:
            self: (todo): write your description
        """
        self.acc += self.toc()

    def release(self):
        """
        Release the next lock.

        Args:
            self: (todo): write your description
        """
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        """
        Reset the internal state.

        Args:
            self: (todo): write your description
        """
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        """
        Initialize the directory.

        Args:
            self: (todo): write your description
        """
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = 'experiment/' + args.save
        else:
            self.dir = 'experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            """
            Create a directory if it doesn t exist.

            Args:
                path: (str): write your description
            """
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        """
        Saves the model to disk.

        Args:
            self: (todo): write your description
            trainer: (todo): write your description
            epoch: (int): write your description
            is_best: (bool): write your description
        """
        trainer.model.save(self.dir, epoch, self.args.model, is_best=is_best)
        # trainer.model_NLEst.save(self.dir, epoch, 'NL_EST', is_best=is_best)
        # trainer.model_KMEst.save(self.dir, epoch, 'KM_EST', is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        """
        Add a log

        Args:
            self: (todo): write your description
            log: (todo): write your description
        """
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        """
        Write log to file.

        Args:
            self: (todo): write your description
            log: (todo): write your description
            refresh: (bool): write your description
        """
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        """
        Called when the request.

        Args:
            self: (todo): write your description
        """
        self.log_file.close()

    def plot_psnr(self, epoch):
        """
        This function plots of the model.

        Args:
            self: (todo): write your description
            epoch: (int): write your description
        """
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, idx, scale):
        """
        Save results to a file.

        Args:
            self: (todo): write your description
            filename: (str): write your description
            save_list: (list): write your description
            idx: (str): write your description
            scale: (float): write your description
        """
        filename = '{}/results/{}_x{}_{}'.format(self.dir, filename, scale, idx)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            #print(normalized.size())
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            #print(ndarr.shape)
            
            misc.imsave('{}{}.png'.format(filename, p), np.squeeze(ndarr))

def quantize(img, rgb_range):
    """
    Quantize an rgb image.

    Args:
        img: (array): write your description
        rgb_range: (array): write your description
    """
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    """
    Calculate the psnr noise.

    Args:
        sr: (array): write your description
        hr: (array): write your description
        scale: (float): write your description
        rgb_range: (todo): write your description
        benchmark: (array): write your description
    """
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, my_model):
    """
    Make an optimizer.

    Args:
        my_model: (todo): write your description
    """
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    """
    Creates a scheduler.

    Args:
        my_optimizer: (todo): write your description
    """
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

