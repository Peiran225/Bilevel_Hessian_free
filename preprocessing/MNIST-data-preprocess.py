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

import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--num_class', type=int, default=10, help="The number of classes")
parser.add_argument('--num_sample', type=int, default=40000, help="Total Number of samples")
parser.add_argument('--num_val_sample', type=int, default=5000, help="Total Number of samples")
parser.add_argument('--noise_rate', type=float, default=0.6, help="Total Number of samples")

args = parser.parse_args()
seed(1)

num_sample = args.num_sample
noise_sample = int(args.num_sample * args.noise_rate)
num_val_sample = args.num_val_sample
num_class = args.num_class

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

trainset = torchvision.datasets.MNIST(
    root='./data/MNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset), shuffle=True, num_workers=2)

images, labels = next(iter(trainloader))
images, labels = np.array(images), np.array(labels)


x = images[:num_sample]; y = labels[:num_sample]
label = np.random.randint(0,num_class,noise_sample)
y_index = np.random.choice(num_sample,noise_sample, replace=False)
for index,lb in zip(y_index, label):
    y[index] = lb
devx = images[num_sample:num_sample + num_val_sample]
devy = labels[num_sample:num_sample + num_val_sample]

testset = torchvision.datasets.MNIST(
    root='./data/MNIST', train=False, download = True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset), shuffle=False, num_workers=2)
tx, ty = next(iter(testloader))
tx, ty = np.array(tx), np.array(ty)

prefix =  '/ocean/projects/cis220038p/junyili/AdaBilevel'
np.save(prefix + '/processed_data/MNIST_data/trainx_' + str(args.noise_rate),x)
np.save(prefix + '/processed_data/MNIST_data/trainy_' + str(args.noise_rate),y)
np.save(prefix + '/processed_data/MNIST_data/y_index_'+ str(args.noise_rate),y_index)
np.save(prefix + '/processed_data/MNIST_data/devx',devx)
np.save(prefix + '/processed_data/MNIST_data/devy',devy)
np.save(prefix + '/processed_data/MNIST_data/testx',tx)
np.save(prefix + '/processed_data/MNIST_data/testy',ty)
