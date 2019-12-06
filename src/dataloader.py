import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from skimage import io

import numpy as np

from com_utlis import *

def data_loader(dataset, batchsize=64):
    datapath = os.path.join("..", "data", dataset)
    files = os.listdir(datapath)

    x = []
    for file in files:
        if file in STOP_FILES:
            continue
        filepath = os.path.join(datapath, file)
        img = io.imread(filepath)
        x.append(img)

    # x = np.array(x)
    x = torch.Tensor(x)
    train_data = torch.utils.data.TensorDataset(x)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)

    return train_dataloader


if __name__ == "__main__":
    loader = data_loader("64x64", batchsize=28)

    for x in loader:
        print(len(x), x[0].shape)


