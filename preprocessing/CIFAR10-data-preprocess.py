import numpy as np
from numpy.random import seed
import os
import json
import matplotlib.image as mpimg
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import torchvision
import torchvision.transforms as transforms
import torch

import pdb


seed(1)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data/CIFAR10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset), shuffle=True, num_workers=2)

images, labels = next(iter(trainloader))
images, labels = np.array(images), np.array(labels)

x = images[:5000]; y = labels[:5000]
label = np.random.randint(0,10,3000)
y_index = np.random.choice(5000,3000, replace=False)
for index,lb in zip(y_index, label):
    y[index] = lb
# pdb.set_trace()
devx = images[5000:6000]
devy = labels[5000:6000]

testset = torchvision.datasets.CIFAR10(
    root='./data/CIFAR10', train=False, download = True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset), shuffle=False, num_workers=2)
tx, ty = next(iter(testloader))
tx, ty = np.array(tx), np.array(ty)

np.save('./CIFAR10_data/trainx',x)
np.save('./CIFAR10_data/trainy',y)
np.save('./CIFAR10_data/devx',devx)
np.save('./CIFAR10_data/devy',devy)
np.save('./CIFAR10_data/testx',tx)
np.save('./CIFAR10_data/testy',ty)
np.save('./CIFAR10_data/y_index',y_index)
