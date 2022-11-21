# from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
# from matplotlib import pyplot as plt


import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
import os

import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import torch

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from data import get_dataloader
from models import model_dict
import os
from utils import AverageMeter, accuracy
import numpy as np
from datetime import datetime
import torch.nn as nn
from torch.autograd import Variable
import copy

import pickle
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 128)')

parser.add_argument('--T', type=float, default=4.0)  # temperature
parser.add_argument('--model_names', type=str, default='resnet20')


parser.add_argument('--alpha', type=float, default=0.5)  # weight for ce and kl

parser.add_argument('--root', type=str, default='dataset')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epoch', type=int, default=240)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])
parser.add_argument('--thresh_hold', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-id', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=1)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_cifar(num_classes=10, dataset_dir='./data', batch_size=10000, crop=False):

    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    simple_transform = transforms.Compose([transforms.ToTensor(),normalize])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    if num_classes == 100:
        trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True,
                                                 download=True, transform=train_transform)

        testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False,
                                                download=True, transform=simple_transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                                download=True, transform=train_transform)

        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                               download=True, transform=simple_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4,
                                              pin_memory=True, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4,
                                             pin_memory=True, shuffle=False)
    return trainloader, testloader

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range

def tsne_plot(save_dir, output, target):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(n_components=2).fit_transform(output)
    tx = tsne[:,0]
    ty = tsne[:,1]
    plt.figure(figsize=(10, 10))
    plt.axis('off')

    plt.scatter(x=tx, y=ty, c=target, cmap=plt.cm.get_cmap("jet", 100), marker='.')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    # #
    plt.tight_layout()
    save_dir = "./"
    plt.savefig(os.path.join(save_dir, 'resnet20_ours.png'), bbox_inches='tight')

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def gen_features(dataloader=None,net=None):
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda()
            # targets = targets.to(device)
            # targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            tsne_plot(args.save_dir, outputs.data.cpu(), targets)
            break

            # visualize(outputs, color=targets)
    return outputs_list, targets_list

if __name__ == '__main__':
    train_loarder, test= get_cifar()
    model = model_dict[args.model_names](num_classes=10)
    ckpt = "/home/cvpr/sdb1/Online-Knowledge-Distillation-via-Collaborative-Learning/experiments/resnet26_resnet20_resnet14_resnet8/2021-12-09-01-33/gpu_num:2/resnet20_1/ckpt/best.pth"
    model_ckpt = torch.load(ckpt)
    model.load_state_dict(model_ckpt['model'])
    model = model.cuda()
    output,targets = gen_features(train_loarder,model)
    # tsne_plot(args.save_dir, output, targets)