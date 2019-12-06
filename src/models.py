import torch.nn as nn

from com_utlis import *


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0), )+self.shape)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, device, no_bn=False, fc=False):
        super(Generator, self).__init__()
        if fc:
            self.main = nn.Sequential(
                Reshape(nz),
                nn.Linear(nz, ngf * 8, bias=True),
                nn.ReLU(True),
                nn.Linear(ngf * 8, ngf * 8, bias=True),
                nn.ReLU(True),
                nn.Linear(ngf * 8, ngf * 16, bias=True),
                nn.ReLU(True),
                nn.Linear(ngf * 16, ngf * 32, bias=True),
                nn.ReLU(True),
                nn.Linear(ngf * 32, nc * 64 * 64, bias=True),
                nn.Tanh(),
                Reshape(nc, 64, 64)
            )
        elif no_bn:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

    def forward(self, inputs):
        return self.main(inputs)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, device, no_bn=False, fc=False, model='wgangp'):
        super(Discriminator, self).__init__()
        if fc and model in ['wgangp', 'wgan']:
            self.main = nn.Sequential(
                Reshape(nc * 64 * 64),
                nn.Linear(nc * 64 * 64, ndf * 32, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 32, ndf * 16, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 16, ndf * 8, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 8, ndf * 4, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 4, 1, bias=True)
            )
        elif fc and model in ['gan']:
            self.main = nn.Sequential(
                Reshape(nc * 64 * 64),
                nn.Linear(nc * 64 * 64, ndf * 32, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 32, ndf * 16, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 16, ndf * 8, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 8, ndf * 4, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(ndf * 4, 1, bias=True),
                nn.Sigmoid()
            )
        elif model in ['gan']:
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif model in ['wgangp', 'wgan']:
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            )

    def forward(self, inputs):
        return self.main(inputs).view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
