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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

trainset = torchvision.datasets.SVHN(
    root='./data/SVHN', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset), shuffle=True, num_workers=2)

images, labels = next(iter(trainloader))
images, labels = np.array(images), np.array(labels)
print(images.shape, labels.shape)
pdb.set_trace()
x = images[:5000]; y = labels[:5000]
label = np.random.randint(0,10,4000)
y_index = np.random.choice(5000,4000, replace=False)
for index,lb in zip(y_index, label):
    y[index] = lb
# pdb.set_trace()
devx = images[5000:10000]
devy = labels[5000:10000]

testset = torchvision.datasets.SVHN(
    root='./data/SVHN', split='test', download = True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset), shuffle=False, num_workers=2)
tx, ty = next(iter(testloader))
tx, ty = np.array(tx), np.array(ty)

np.save('./SVHN_data/trainx',x)
np.save('./SVHN_data/trainy',y)
np.save('./SVHN_data/devx',devx)
np.save('./SVHN_data/devy',devy)
np.save('./SVHN_data/testx',tx)
np.save('./SVHN_data/testy',ty)
np.save('./SVHN_data/y_index',y_index)
