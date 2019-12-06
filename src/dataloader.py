import os

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from com_utlis import *

def data_loader(dataset, batch_size=64, img_size=64):
    datapath = os.path.join("..", "data", dataset)
    dataset = dset.ImageFolder(root=datapath,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == "__main__":
    loader = data_loader("64x64", batchsize=28)

    for x in loader:
        print(len(x), x[0].shape)


