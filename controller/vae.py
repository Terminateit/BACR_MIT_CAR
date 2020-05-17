'''Imports'''
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
import random
random.seed(350)
import numpy as np
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import os
import csv
from torch.distributions.normal import Normal
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from functools import partial
from os.path import join, exists
from os import mkdir
from tqdm import tqdm
import math
import gym
import gym.envs.box2d

# Make train results reproducible
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = torch.cuda.is_available()
device_use = torch.device("cuda" if use_cuda else "cpu")

#################################################################################################################

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class ConvVAE(nn.Module):
    def __init__(self, image_channels = 3, hidden_dim = 1024, latent_dim=32, kl_tolerance = 0.5):

        '''
        Args:
            image_channels: A integer indicating the number of channels in image.
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
        '''
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.kl_tolerance = kl_tolerance

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=4, stride = 2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=32,out_channels= 64, kernel_size = 4, stride = 2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64,out_channels= 128, kernel_size = 4, stride = 2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128,out_channels= 256, kernel_size = 4, stride = 2),
            nn.ReLU(True),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=hidden_dim,out_channels= 128, kernel_size = 5, stride = 2),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(in_channels=128,out_channels= 64, kernel_size = 5, stride = 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=64,out_channels= 32, kernel_size = 6, stride = 2),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(in_channels=32,out_channels= image_channels, kernel_size = 6, stride = 2),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.fc1(conv)
        return h1, h1

    def decode(self, z):
        
        deconv_input = self.fc3(z)
        deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        sigma = logvar.exp()
        eps = torch.randn_like(sigma)
        return  eps.mul(sigma).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar, z  

    def architecture(self):
        return ConvVAE().to(self.used_device)

    def loss(self, decoded, x, mu, logsigma):
        # BCE =  F.binary_cross_entropy(decoded, x, reduction='sum')
        BCE = torch.sum(torch.square((decoded-x)))
        reconstruct_loss = BCE
        KLD = -0.5 * torch.sum((1 + 2*logsigma - mu**2 - (2*logsigma).exp()))
        KLD = torch.max(KLD, torch.Tensor([self.kl_tolerance * self.latent_dim]).cuda())
        KLD = torch.mean(KLD)
        regularization_loss = KLD
        loss = reconstruct_loss + regularization_loss
        return loss
