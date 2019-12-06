import os
import random
import argparse
import time

from tqdm import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from com_utlis import *
from models import *
from dataloader import *
from train_model import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='| 64x64 |')
parser.add_argument('--model', type=str, required=True, help='| wgangp | wgan | gan |')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--imagesize', type=int, default=64, help='input image size (height equals to width)')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='generator kernel size')
parser.add_argument('--ndf', type=int, default=64, help='')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument("--n_step", type=int, default=4, help="number of training steps for discriminator per iter")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--nc', type=int, default=3, help='# input channels')
parser.add_argument('--no_bn', type=bool, default=False, action='store_true',
                    help='disable batch norm in generator')
parser.add_argument('--fc', type=bool, default=False, action='store_true',
                    help='enable MLP generator and MLP discriminator')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":

    try:
      os.makedirs(opt.outf)
    except OSError:
      pass


    if opt.seed is None:
        seed = time.time()
    else:
        seed = opt.seed
    print("Seed: ", seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if not torch.cuda.is_available() and opt.cuda:
        raise Exception("ERROR: no CUDA device.")
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    nz = opt.nz
    ndf = opt.ndf
    ngf = opt.ngf
    nc = opt.nc

    loader = data_loader(opt.dataset, batch_size=opt.batchsize, img_size=opt.imagesize)
    print("Data loading finished.")


    netG = Generator(nz, ngf, nc, device, no_bn=opt.no_bn, fc=opt.fc).to(device)
    netG.apply(weights_init)
    netD = Discriminator(nc, ndf, device, no_bn=opt.no_bn, fc=opt.fc, model=opt.model).to(device)
    netD.apply(weights_init)

    if opt.model in ["wgangp", 'gan']:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    elif opt.model == "wgan":
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)

    train(netD, netG, optimizerD, optimizerG, loader, opt.niter, opt.model, device, nz, wclamp=0.01, n_step=opt.n_step)



