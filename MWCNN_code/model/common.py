import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    """
    Default convolution layer.

    Args:
        in_channels: (int): write your description
        out_channels: (int): write your description
        kernel_size: (int): write your description
        bias: (todo): write your description
        dilation: (str): write your description
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)+dilation-1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):
    """
    Default conv2d conv2d conv2d layer.

    Args:
        in_channels: (int): write your description
        out_channels: (int): write your description
        kernel_size: (int): write your description
        bias: (todo): write your description
        groups: (array): write your description
    """
    return nn.Conv2d(
        in_channels,out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups=groups)

#def shuffle_channel()

def channel_shuffle(x, groups):
    """
    Shuffle channel.

    Args:
        x: (todo): write your description
        groups: (array): write your description
    """
    batchsize, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def pixel_down_shuffle(x, downsacale_factor):
    """
    Shuffle a pixel pixel.

    Args:
        x: (todo): write your description
        downsacale_factor: (float): write your description
    """
    batchsize, num_channels, height, width = x.size()

    out_height = height // downsacale_factor
    out_width = width // downsacale_factor
    input_view = x.contiguous().view(batchsize, num_channels, out_height, downsacale_factor, out_width,
                                     downsacale_factor)

    num_channels *= downsacale_factor ** 2
    unshuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()

    return unshuffle_out.view(batchsize, num_channels, out_height, out_width)



def sp_init(x):
    """
    Initialize sparsity.

    Args:
        x: (todo): write your description
    """

    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x_LL = x01[:, :, :, 0::2]
    x_HL = x02[:, :, :, 0::2]
    x_LH = x01[:, :, :, 1::2]
    x_HH = x02[:, :, :, 1::2]


    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def dwt_init(x):
    """
    Initialize dwt.

    Args:
        x: (todo): write your description
    """

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    """
    Initialize the init.

    Args:
        x: (todo): write your description
    """
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class Channel_Shuffle(nn.Module):
    def __init__(self, conv_groups):
        """
        Initialize groups.

        Args:
            self: (todo): write your description
            conv_groups: (todo): write your description
        """
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        """
        R forward forward forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return channel_shuffle(x, self.conv_groups)

class SP(nn.Module):
    def __init__(self):
        """
        Initialize gradient

        Args:
            self: (todo): write your description
        """
        super(SP, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        """
        Perform forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return sp_init(x)

class Pixel_Down_Shuffle(nn.Module):
    def __init__(self):
        """
        Initialize the gradients.

        Args:
            self: (todo): write your description
        """
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        """
        Perform forward forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return pixel_down_shuffle(x, 2)

class DWT(nn.Module):
    def __init__(self):
        """
        Initialize gradient

        Args:
            self: (todo): write your description
        """
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        """
        Forward forward forward forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        """
        Initialize the gradient

        Args:
            self: (todo): write your description
        """
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return iwt_init(x)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        """
        Initialize the rgb.

        Args:
            self: (todo): write your description
            rgb_range: (float): write your description
            rgb_mean: (todo): write your description
            rgb_std: (str): write your description
            sign: (float): write your description
        """
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.create_graph = False
            self.volatile = True
class MeanShift2(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        """
        Initialize the rgb.

        Args:
            self: (todo): write your description
            rgb_range: (float): write your description
            rgb_mean: (todo): write your description
            rgb_std: (str): write your description
            sign: (float): write your description
        """
        super(MeanShift2, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1)
        self.weight.data.div_(std.view(4, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
        if sign==-1:
            self.volatile = True

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=False, act=nn.ReLU(True)):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            stride: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
        """

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class BBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward computation for x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x).mul(self.res_scale)
        return x

class DBlock_com(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Evaluate of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x)
        return x

class DBlock_inv(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Evaluate of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x)
        return x

class DBlock_com1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Evaluate of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x)
        return x

class DBlock_inv1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Evaluate of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x)
        return x

class DBlock_com2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Evaluate of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x)
        return x

class DBlock_inv2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Evaluate of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x)
        return x

class ShuffleBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,conv_groups=1):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
            conv_groups: (todo): write your description
        """

        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward computation for x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):
    def __init__(
        self, conv, conv1, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize the convolution layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            conv1: (todo): write your description
            in_channels: (int): write your description
            out_channels: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward computation for x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.body(x).mul(self.res_scale)
        return x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize the convolution.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            n_feat: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Block(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Initialize the convolutional layer.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            n_feat: (int): write your description
            kernel_size: (int): write your description
            bias: (float): write your description
            bn: (int): write your description
            act: (str): write your description
            nn: (todo): write your description
            ReLU: (todo): write your description
            res_scale: (float): write your description
        """

        super(Block, self).__init__()
        m = []
        for i in range(4):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward function todo.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        res = self.body(x).mul(self.res_scale)
        # res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        """
        Initialize a batch.

        Args:
            self: (todo): write your description
            conv: (todo): write your description
            scale: (float): write your description
            n_feat: (int): write your description
            bn: (int): write your description
            act: (str): write your description
            bias: (float): write your description
        """

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




