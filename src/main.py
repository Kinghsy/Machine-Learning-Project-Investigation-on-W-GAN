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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='| 64x64 |')
parser.add_argument('--loss', type=str, required=True, help='| wgangp | wgan | gan |')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--imagesize', type=int, default=64, help='input image size (height equals to width)')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='temp_floder', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--nc', type=int, default=3, help='# input channels')

opt = parser.parse_args()
print(opt)


def calculate_gradient_penatly(netD, real_imgs, fake_imgs):
    eta = torch.FloatTensor(real_imgs.size(0), 1, 1, 1).uniform_(0, 1).to(device)
    eta = eta.expand(real_imgs.size(0), real_imgs.size(1), real_imgs.size(2), real_imgs.size(3)).to(device)

    interpolated = eta * real_imgs + ((1 - eta) * fake_imgs)
    interpolated.to(device)

    # define it to calculate gradient
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # calculate probaility of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradients_penalty


def train(opt, loader, device, nz, netD, netG, optimizerD, optimizerG):

    step = 0
    for epoch in tqdm(range(opt.niter)):
        for i, x in enumerate(loader):

            real_imgs = x[0].permute(0, 3, 1, 2).to(device)
            batch_size = real_imgs.size(0)

            # -----------------
            #  Train Discriminator
            # -----------------

            netD.zero_grad()
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_imgs = netG(noise)


            real_validity = netD(real_imgs)
            fake_validity = netD(fake_imgs)
            gradient_penalty = calculate_gradient_penatly(netD, real_imgs.data, fake_imgs.data)

            # Loss measures generator's ability to fool the discriminator
            errD = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

            errD.backward()
            optimizerD.step()

            optimizerG.zero_grad()

            # Train the generator every n_critic iterations
            if step % opt.n_critic == 0:
                # ---------------------
                #  Train Generator
                # ---------------------

                # Generate a batch of images
                fake_imgs = netG(noise)
                # Adversarial loss
                errG = -torch.mean(netD(fake_imgs))

                errG.backward()
                optimizerG.step()

            print(f'[{epoch + 1}/{opt.niter}][{i}/{len(loader)}] '
                  f'Loss_D: {errD.item():.4f} '
                  f'Loss_G: {errG.item():.4f}.')

            if epoch % 1 == 0:
                vutils.save_image(real_imgs,
                                  f'{opt.outf}/real_samples.png',
                                  normalize=True)
                vutils.save_image(netG(noise).detach(),
                                  f'{opt.outf}/fake_samples_epoch_{epoch}.png',
                                  normalize=True)
            step += 1

        # do checkpointing
        torch.save(netG, f'{opt.outf}/netG_epoch_{epoch + 1}.pth')
        torch.save(netD, f'{opt.outf}/netD_epoch_{epoch + 1}.pth')


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


    loader = data_loader(opt.dataset, batchsize=opt.batchsize)
    print("Data loading finished.")

    netG = Generator(nz, ngf, nc, device).to(device)
    netG.apply(weights_init)
    netD = Discriminator(nc, ndf, device).to(device)
    netD.apply(weights_init)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))


    train(opt, loader, device, nz, netD, netG, optimizerD, optimizerG)



