import os
import json
import numpy as np

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

record = {
    "lossD": [],
    "lossG": [],
    "gp": []
}

def calculate_gradient_penatly(netD, real_imgs, fake_imgs, device):
    eta = torch.FloatTensor(real_imgs.size(0), 1, 1, 1).uniform_(0, 1).to(device)
    eta = eta.expand(real_imgs.size(0), real_imgs.size(1), real_imgs.size(2), real_imgs.size(3)).to(device)

    interpolated = (eta * real_imgs + ((1 - eta) * fake_imgs)).to(device)
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
    prob_interpolated = netD(interpolated)

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


def train(netD, netG, optimD, optimG, data_loader, n_iteration, model, device, nz, wclamp=0.01, n_step=4):
    for epoch in range(n_iteration):
        lossD = []
        lossG = []
        gp = []
        for i, (real_imgs, _) in enumerate(data_loader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            netD.zero_grad()
            if model == "wgan":
                for p in netD.parameters():
                    p.data.clamp_(-wclamp, wclamp)

            noise = (torch.randn(batch_size, nz, 1, 1)).to(device)
            fake_imgs = netG(noise)

            real_validity = netD(real_imgs)
            fake_validity = netG(fake_imgs)

            if model == 'wgangp':
                gp = calculate_gradient_penatly(netD, real_imgs, fake_imgs, device)
                errD = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                lossD.append((-torch.mean(real_validity) + torch.mean(fake_validity)).item())
                gp.append(gp.item())
            elif model == 'wgan':
                errD = -torch.mean(real_validity) + torch.mean(fake_validity)
                lossD.append(errD.item())
            elif model == 'gan':
                loss = nn.BCELoss().to(device)
                real_label = torch.autograd.Variable(torch.ones(batch_size)).to(device)
                fake_label = torch.autograd.Variable(torch.zeros(batch_size)).to(device)
                errD = loss(real_validity, real_label) + loss(fake_validity, fake_label)
                lossD.append(errD.item())

            errD.backward()
            optimD.step()

            netG.zero_grad()
            if i % n_step == 0:
                noise = (torch.randn(batch_size, nz, 1, 1)).to(device)
                fake_imgs = netG(noise)
                fake_validity = netD(fake_imgs)
                if model == 'gan':
                    loss = nn.BCELoss()
                    real_label = torch.autograd.Variable(torch.ones(batch_size)).to(device)
                    errG = loss(fake_validity, real_label)
                else:
                    errG = -torch.mean(fake_validity)
                lossG.append(errG.item())
                errG.backward()
                optimG.step()

            print("[{}/{}][{}/{}] D_loss: {}, G_loss: {}".format(
                epoch + 1, n_iteration, i+1, len(data_loader), errD.item(), errG.item()
            ))

            if i % 10 == 0:
                vutils.save_image(real_imgs, os.path.join(
                    output_data_path, model, "real_{}.png".format(epoch)
                ))
                vutils.save_image(netG(noise).detach(), os.path.join(
                    output_data_path, model, "fake_{}.png".format(epoch)
                ))

        record["lossD"].append(np.mean(lossD))
        record["lossG"].append(np.mean(lossG))
        record["gp"].append(np.mean(gp))
        s = json.dumps(record)
        with open(os.path.join(output_data_path, model, "loss.log"), 'w') as f:
            f.write(s)

        if epoch % 10 == 0:
            torch.save(netG, os.path.join(output_data_path, model, "G_{}.svs".format(epoch)))
            torch.save(netD, os.path.join(output_data_path, model, "D_{}.svs".format(epoch)))




