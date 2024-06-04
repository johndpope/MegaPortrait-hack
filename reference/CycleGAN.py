import torch
import torch.nn as nn
import torch.nn.functional as F


class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class DECNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Deconv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)


class ResBlock(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm='inorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []

        # 1st conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        # 2nd conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=[])]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)


class CNR1d(nn.Module):
    def __init__(self, nch_in, nch_out, norm='bnorm', relu=0.0, drop=[]):
        super().__init__()

        if norm == 'bnorm':
            bias = False
        else:
            bias = True

        layers = []
        layers += [nn.Linear(nch_in, nch_out, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Deconv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

        # layers = [nn.Upsample(scale_factor=2, mode='bilinear'),
        #           nn.ReflectionPad2d(1),
        #           nn.Conv2d(nch_in , nch_out, kernel_size=3, stride=1, padding=0)]
        #
        # self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(x)


class Linear(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x):
        return self.linear(x)


class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class Padding(nn.Module):
    def __init__(self, padding, padding_mode='zeros', value=0):
        super(Padding, self).__init__()
        if padding_mode == 'reflection':
            self. padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)


class Pooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='avg'):
        super().__init__()

        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='nearest'):
        super().__init__()

        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest', align_corners=True)
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])

        return torch.cat([x2, x1], dim=1)


class TV1dLoss(nn.Module):
    def __init__(self):
        super(TV1dLoss, self).__init__()

    def forward(self, input):
        # loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
        #        torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        loss = torch.mean(torch.abs(input[:, :-1] - input[:, 1:]))

        return loss


class TV2dLoss(nn.Module):
    def __init__(self):
        super(TV2dLoss, self).__init__()

    def forward(self, input):
        loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
               torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return loss


class SSIM2dLoss(nn.Module):
    def __init__(self):
        super(SSIM2dLoss, self).__init__()

    def forward(self, input, targer):
        loss = 0
        return loss


import argparse

import torch.backends.cudnn as cudnn
from train import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
parser = argparse.ArgumentParser(description='Train the CycleGAN network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='cyclegan', dest='scope')
parser.add_argument('--norm', type=str, default='inorm', dest='norm')
parser.add_argument('--name_data', type=str, default='monet2photo', dest='name_data')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')

parser.add_argument('--dir_data', default='../datasets', dest='dir_data')
parser.add_argument('--dir_result', default='./results', dest='dir_result')

parser.add_argument('--num_epoch', type=int,  default=300, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=4, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=2e-4, dest='lr_G')
parser.add_argument('--lr_D', type=float, default=2e-4, dest='lr_D')

parser.add_argument('--num_freq_disp', type=int,  default=50, dest='num_freq_disp')
parser.add_argument('--num_freq_save', type=int,  default=10, dest='num_freq_save')

parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step', 'plateau', 'cosine'], dest='lr_policy')
parser.add_argument('--n_epochs', type=int, default=100, dest='n_epochs')
parser.add_argument('--n_epochs_decay', type=int, default=100, dest='n_epochs_decay')
parser.add_argument('--lr_decay_iters', type=int, default=50, dest='lr_decay_iters')

parser.add_argument('--wgt_c_a', type=float, default=1e1, dest='wgt_c_a')
parser.add_argument('--wgt_c_b', type=float, default=1e1, dest='wgt_c_b')
parser.add_argument('--wgt_i', type=float, default=5e-1, dest='wgt_i')
# parser.add_argument('--wgt_i', type=float, default=0e-1, dest='wgt_i')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--ny_in', type=int, default=256, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=256, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=3, dest='nch_in')

parser.add_argument('--ny_load', type=int, default=286, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=286, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=3, dest='nch_load')

parser.add_argument('--ny_out', type=int, default=256, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=256, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=3, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')
parser.add_argument('--direction', default='A2B', dest='direction')

parser.add_argument('--nblk', type=int, default=6, dest='nblk')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'test':
        TRAINER.test()

if __name__ == '__main__':
    main()
import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

dir_result = './results/cyclegan/monet2photo/images'
lst_result = os.listdir(dir_result)

nx = 256
ny = 256
nch = 3

n = 8
m = 6
m_id = [0, 3, 1, 2]

n_id = np.arange(len(lst_result)//m)
np.random.shuffle(n_id)


## From domain A to domain B
img = torch.zeros((n*(m-4), ny, nx, nch))

for i in range(m-4):
    for j in range(n):
        p = m_id[i + 0] + m * n_id[j]
        # p = m_id[i + 2] + m * n_id[j]
        q = n*i + j

        img[q, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_result[p]))[:, :, :nch])

img = img.permute((0, 3, 1, 2))

plt.figure(figsize=(n, (m-4)))
plt.axis("off")
# plt.title("Generated Images")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))

plt.show()

## From domain B to domain A
img = torch.zeros((n*(m-4), ny, nx, nch))

for i in range(m-4):
    for j in range(n):
        # p = m_id[i + 0] + m * n_id[j]
        p = m_id[i + 2] + m * n_id[j]
        q = n*i + j

        img[q, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_result[p]))[:, :, :nch])

img = img.permute((0, 3, 1, 2))

plt.figure(figsize=(n, (m-4)))
plt.axis("off")
# plt.title("Generated Images")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True), (1, 2, 0)))

plt.show()

from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc5 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc6 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc7 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.2, drop=[])
        self.enc8 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])

        self.dec8 = DECNR2d(1 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec7 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec6 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=0.5)
        self.dec5 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec4 = DECNR2d(2 * 8 * self.nch_ker, 4 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec3 = DECNR2d(2 * 4 * self.nch_ker, 2 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec2 = DECNR2d(2 * 2 * self.nch_ker, 1 * self.nch_ker, stride=2, norm=self.norm, relu=0.0, drop=[])
        self.dec1 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_out, stride=2, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec8 = self.dec8(enc8)
        dec7 = self.dec7(torch.cat([enc7, dec8], dim=1))
        dec6 = self.dec6(torch.cat([enc6, dec7], dim=1))
        dec5 = self.dec5(torch.cat([enc5, dec6], dim=1))
        dec4 = self.dec4(torch.cat([enc4, dec5], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

        x = torch.tanh(dec1)

        return x



class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=6):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in,      1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)

        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]

            self.res = nn.Sequential(*res)

        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        if self.nblk:
            x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 32 x 32 x 512
        # dsc5 : 32 x 32 x 512 -> 32 x 32 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


from __future__ import absolute_import, division, print_function

import os
import logging
import torch
# import argparse

''''
class Logger:
class Parser:
'''
class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        str_ids = self.__args.gpu_ids.split(',')
        self.__args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.__args.gpu_ids.append(id)
        # if len(self.__args.gpu_ids) > 0:
        #     torch.cuda.set_device(self.__args.gpu_ids[0])

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['dir_log'], params_dict['scope'], params_dict['name_data'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)


class Logger:
    def __init__(self, info=logging.INFO, name=__name__):
        logger = logging.getLogger(name)
        logger.setLevel(info)

        self.__logger = logger

    def get_logger(self, handler_type='stream_handler'):
        if handler_type == 'stream_handler':
            handler = logging.StreamHandler()
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(log_format)
        else:
            handler = logging.FileHandler('utils.log')

        self.__logger.addHandler(handler)

        return self.__logger


from model import *
from dataset import *

import itertools
from statistics import mean

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_c_a = args.wgt_c_a
        self.wgt_c_b = args.wgt_c_b
        self.wgt_i = args.wgt_i

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.direction = args.direction
        self.name_data = args.name_data

        self.nblk = args.nblk

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                    'netD_a': netD_a.state_dict(), 'netD_b': netD_b.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG_a2b, netG_b2a, netD_a=[], netD_b=[], optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])
            netD_a.load_state_dict(dict_net['netD_a'])
            netD_b.load_state_dict(dict_net['netD_b'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch

        elif mode == 'test':
            netG_a2b.load_state_dict(dict_net['netG_a2b'])
            netG_b2a.load_state_dict(dict_net['netG_b2a'])

            return netG_a2b, netG_b2a, epoch

    def preprocess(self, data):
        normalize = Normalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(normalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_c_a = self.wgt_c_a
        wgt_c_b = self.wgt_c_b
        wgt_i = self.wgt_i

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')

        transform_train = transforms.Compose([Normalize(), RandomFlip(), Rescale((self.ny_load, self.nx_load)), RandomCrop((self.ny_in, self.nx_in)), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = Dataset(dir_data_train, direction=self.direction, data_type=self.data_type, transform=transform_train)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        num_train = len(dataset_train)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        # netG_a2b = UNet(nch_in, nch_out, nch_ker, norm)
        # netG_b2a = UNet(nch_in, nch_out, nch_ker, norm)
        netG_a2b = ResNet(nch_in, nch_out, nch_ker, norm, nblk=self.nblk)
        netG_b2a = ResNet(nch_in, nch_out, nch_ker, norm, nblk=self.nblk)

        netD_a = Discriminator(nch_in, nch_ker, norm)
        netD_b = Discriminator(nch_in, nch_ker, norm)
        
        init_net(netG_a2b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG_b2a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        init_net(netD_a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD_b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_Cycle = nn.L1Loss().to(device)   # L1
        fn_GAN = nn.BCEWithLogitsLoss().to(device)
        fn_Ident = nn.L1Loss().to(device)   # L1

        paramsG_a2b = netG_a2b.parameters()
        paramsG_b2a = netG_b2a.parameters()
        paramsD_a = netD_a.parameters()
        paramsD_b = netD_b.parameters()

        optimG = torch.optim.Adam(itertools.chain(paramsG_a2b, paramsG_b2a), lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(itertools.chain(paramsD_a, paramsD_b), lr=lr_D, betas=(self.beta1, 0.999))

        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, st_epoch = \
                self.load(dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netD_b.train()

            loss_G_a2b_train = []
            loss_G_b2a_train = []
            loss_D_a_train = []
            loss_D_b_train = []
            loss_C_a_train = []
            loss_C_b_train = []
            loss_I_a_train = []
            loss_I_b_train = []

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input_a = data['dataA'].to(device)
                input_b = data['dataB'].to(device)

                # forward netG
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)

                recon_b = netG_a2b(output_a)
                recon_a = netG_b2a(output_b)

                # backward netD
                set_requires_grad([netD_a, netD_b], True)
                optimD.zero_grad()

                # backward netD_a
                pred_real_a = netD_a(input_a)
                pred_fake_a = netD_a(output_a.detach())

                loss_D_a_real = fn_GAN(pred_real_a, torch.ones_like(pred_real_a))
                loss_D_a_fake = fn_GAN(pred_fake_a, torch.zeros_like(pred_fake_a))
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                # backward netD_b
                pred_real_b = netD_b(input_b)
                pred_fake_b = netD_b(output_b.detach())

                loss_D_b_real = fn_GAN(pred_real_b, torch.ones_like(pred_real_b))
                loss_D_b_fake = fn_GAN(pred_fake_b, torch.zeros_like(pred_fake_b))
                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

                # backward netD
                loss_D = loss_D_a + loss_D_b
                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad([netD_a, netD_b], False)
                optimG.zero_grad()

                if wgt_i > 0:
                    ident_b = netG_a2b(input_b)
                    ident_a = netG_b2a(input_a)

                    loss_I_a = fn_Ident(ident_a, input_a)
                    loss_I_b = fn_Ident(ident_b, input_b)
                else:
                    loss_I_a = 0
                    loss_I_b = 0

                pred_fake_a = netD_a(output_a)
                pred_fake_b = netD_b(output_b)

                loss_G_a2b = fn_GAN(pred_fake_b, torch.ones_like(pred_fake_b))
                loss_G_b2a = fn_GAN(pred_fake_a, torch.ones_like(pred_fake_a))

                loss_C_a = fn_Cycle(input_a, recon_a)
                loss_C_b = fn_Cycle(input_b, recon_b)

                loss_G = (loss_G_a2b + loss_G_b2a) + \
                         (wgt_c_a * loss_C_a + wgt_c_b * loss_C_b) + \
                         (wgt_c_a * loss_I_a + wgt_c_b * loss_I_b) * wgt_i

                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_a2b_train += [loss_G_a2b.item()]
                loss_G_b2a_train += [loss_G_b2a.item()]

                loss_D_a_train += [loss_D_a.item()]
                loss_D_b_train += [loss_D_b.item()]

                loss_C_a_train += [loss_C_a.item()]
                loss_C_b_train += [loss_C_b.item()]

                if wgt_i > 0:
                    loss_I_a_train += [loss_I_a.item()]
                    loss_I_b_train += [loss_I_b.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'G_a2b: %.4f G_b2a: %.4f D_a: %.4f D_b: %.4f C_a: %.4f C_b: %.4f I_a: %.4f I_b: %.4f'
                      % (epoch, i, num_batch_train,
                         mean(loss_G_a2b_train), mean(loss_G_b2a_train),
                         mean(loss_D_a_train), mean(loss_D_b_train),
                         mean(loss_C_a_train), mean(loss_C_b_train),
                         mean(loss_I_a_train), mean(loss_I_b_train)))

                if should(num_freq_disp):
                    ## show output
                    input_a = transform_inv(input_a)
                    output_a = transform_inv(output_a)
                    input_b = transform_inv(input_b)
                    output_b = transform_inv(output_b)

                    writer_train.add_images('input_a', input_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_a', output_a, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('input_b', input_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output_b', output_b, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_G_a2b', mean(loss_G_a2b_train), epoch)
            writer_train.add_scalar('loss_G_b2a', mean(loss_G_b2a_train), epoch)
            writer_train.add_scalar('loss_D_a', mean(loss_D_a_train), epoch)
            writer_train.add_scalar('loss_D_b', mean(loss_D_b_train), epoch)
            writer_train.add_scalar('loss_C_a', mean(loss_C_a_train), epoch)
            writer_train.add_scalar('loss_C_b', mean(loss_C_b_train), epoch)
            writer_train.add_scalar('loss_I_a', mean(loss_I_a_train), epoch)
            writer_train.add_scalar('loss_I_b', mean(loss_I_b_train), epoch)

            # # update schduler
            # # schedG.step()
            # # schedD.step()

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(self.dir_result, self.scope, name_data)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        dir_data_test = os.path.join(self.dir_data, self.name_data, 'test')

        transform_test = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_test = Dataset(dir_data_test, data_type=self.data_type, transform=transform_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        # netG_a2b = UNet(nch_in, nch_out, nch_ker, norm)
        # netG_b2a = UNet(nch_in, nch_out, nch_ker, norm)
        netG_a2b = ResNet(nch_in, nch_out, nch_ker, norm, nblk=self.nblk)
        netG_b2a = ResNet(nch_in, nch_out, nch_ker, norm, nblk=self.nblk)

        init_net(netG_a2b, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netG_b2a, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## load from checkpoints
        st_epoch = 0

        netG_a2b, netG_b2a, st_epoch = self.load(dir_chck, netG_a2b, netG_b2a, mode=mode)

        ## test phase
        with torch.no_grad():
            netG_a2b.eval()
            netG_b2a.eval()
            # netG_a2b.train()
            # netG_b2a.train()

            gen_loss_l1_test = 0
            for i, data in enumerate(loader_test, 1):
                input_a = data['dataA'].to(device)
                input_b = data['dataB'].to(device)

                # forward netG
                output_b = netG_a2b(input_a)
                output_a = netG_b2a(input_b)

                recon_b = netG_a2b(output_a)
                recon_a = netG_b2a(output_b)

                input_a = transform_inv(input_a)
                input_b = transform_inv(input_b)
                output_a = transform_inv(output_a)
                output_b = transform_inv(output_b)
                recon_a = transform_inv(recon_a)
                recon_b = transform_inv(recon_b)

                for j in range(input_a.shape[0]):
                    name = batch_size * (i - 1) + j
                    fileset = {'name': name,
                               'input_a': "%04d-input_a.png" % name,
                               'input_b': "%04d-input_b.png" % name,
                               'output_a': "%04d-output_a.png" % name,
                               'output_b': "%04d-output_b.png" % name,
                               'recon_a': "%04d-recon_a.png" % name,
                               'recon_b': "%04d-recon_b.png" % name}

                    plt.imsave(os.path.join(dir_result_save, fileset['input_a']), input_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['input_b']), input_b[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['output_a']), output_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['output_b']), output_b[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['recon_a']), recon_a[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['recon_b']), recon_b[j, :, :, :].squeeze())

                    append_index(dir_result, fileset)

                    print("%d / %d" % (name + 1, num_test))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)

import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
import os


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, direction='A2B', data_type='float32', nch=3, transform=[]):
        self.data_dir_a = data_dir + 'A'
        self.data_dir_b = data_dir + 'B'
        self.transform = transform
        self.direction = direction
        self.data_type = data_type
        self.nch = nch

        dataA = [f for f in os.listdir(self.data_dir_a) if f.endswith('.jpg')]
        dataA.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        dataB = [f for f in os.listdir(self.data_dir_b) if f.endswith('.jpg')]
        dataB.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.names = (dataA, dataB)

    def __getitem__(self, index):

        # x = np.load(os.path.join(self.data_dir, self.names[0][index]))
        # y = np.load(os.path.join(self.data_dir, self.names[1][index]))

        dataA = plt.imread(os.path.join(self.data_dir_a, self.names[0][index])).squeeze()
        dataB = plt.imread(os.path.join(self.data_dir_b, self.names[1][index])).squeeze()

        if dataA.dtype == np.uint8:
            dataA = dataA / 255.0

        if dataB.dtype == np.uint8:
            dataB = dataB / 255.0

        if len(dataA.shape) == 2:
            dataA = np.expand_dims(dataA, axis=2)
            dataA = np.tile(dataA, (1, 1, 3))
        if len(dataB.shape) == 2:
            dataB = np.expand_dims(dataB, axis=2)
            dataB = np.tile(dataB, (1, 1, 3))

        if self.direction == 'A2B':
            data = {'dataA': dataA, 'dataB': dataB}
        else:
            data = {'dataA': dataB, 'dataB': dataA}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.names[0])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        dataA, dataB = data['dataA'], data['dataB']

        dataA = dataA.transpose((2, 0, 1)).astype(np.float32)
        dataB = dataB.transpose((2, 0, 1)).astype(np.float32)
        return {'dataA': torch.from_numpy(dataA), 'dataB': torch.from_numpy(dataB)}


class Normalize(object):
    def __call__(self, data):
        # Nomalize [0, 1] => [-1, 1]

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data

        dataA, dataB = data['dataA'], data['dataB']
        dataA = 2 * dataA - 1
        dataB = 2 * dataB - 1
        return {'dataA': dataA, 'dataB': dataB}


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        dataA, dataB = data['dataA'], data['dataB']

        if np.random.rand() > 0.5:
            dataA = np.fliplr(dataA)
            dataB = np.fliplr(dataB)

        # if np.random.rand() > 0.5:
        #     dataA = np.flipud(dataA)
        #     dataB = np.flipud(dataB)

        return {'dataA': dataA, 'dataB': dataB}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    dataA = transform.resize(dataA, (new_h, new_w))
    dataB = transform.resize(dataB, (new_h, new_w))

    return {'dataA': dataA, 'dataB': dataB}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    dataA = dataA[top: top + new_h, left: left + new_w]
    dataB = dataB[top: top + new_h, left: left + new_w]

    return {'dataA': dataA, 'dataB': dataB}


class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}


class Denomalize(object):
    def __call__(self, data):
        # Denomalize [-1, 1] => [0, 1]

        # for key, value in data:
        #     data[key] = (value + 1) / 2 * 255
        #
        # return data

        return (data + 1) / 2

        # input, label = data['input'], data['label']
        # input = (input + 1) / 2 * 255
        # label = (label + 1) / 2 * 255
        # return {'input': input, 'label': label}

