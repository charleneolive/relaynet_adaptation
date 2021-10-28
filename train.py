import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torch
import pathlib
import pandas as pd
from tqdm import tqdm
import random
from random import randint
from random import shuffle
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
from torch.autograd import Variable
from torch.optim import lr_scheduler

from networks.data_utils import get_imdb_data
from networks.relay_net import ReLayNet
from networks.solver import Solver
from networks.net_api.losses import DiceLoss, CrossEntropyLoss2d

class RandomTransforms(object):
    def __init__(self, height, width, layers, prob1=1, prob2=0.5, border = [8,8,8,8]):
        # tblr
        self.height = height
        self.width = width
        self.layers = layers
        self.prob1 = prob1
        self.prob2 = prob2
        self.b = border

    def __call__(self, image, target, weight):

        if random.random()<self.prob1:
            pad_image = np.zeros((self.height+self.b[0]+self.b[1], self.width+self.b[2]+self.b[3], 1))
            pad_target = np.zeros((self.height+self.b[0]+self.b[1], self.width+self.b[2]+self.b[3], self.layers))
            pad_weight = np.zeros((self.height+self.b[0]+self.b[1], self.width+self.b[2]+self.b[3]))
            
            # mirror image
            pad_image[self.b[0]:-self.b[1],self.b[2]:-self.b[3],:] = image 
            pad_image[:self.b[0],self.b[2]:-self.b[3]] = image[self.b[0]-1:None:-1]# fill up top border
            pad_image[-self.b[1]:,self.b[2]:-self.b[3]] = image[:-self.b[1]-1:-1] # fill up bottom border
            pad_image[:,self.b[2]-1:None:-1] = pad_image[:,self.b[2]:2*self.b[2]] # fill up left border
            pad_image[:,-self.b[3]:] = pad_image[:,-self.b[3]-1:-2*self.b[3]-1:-1] # fill up right border

            # mirror target
            pad_target[self.b[0]:-self.b[1],self.b[2]:-self.b[3],:] = target 
            pad_target[:self.b[0],self.b[2]:-self.b[3]] = target[self.b[0]-1:None:-1]# fill up top border
            pad_target[-self.b[1]:,self.b[2]:-self.b[3]] = target[:-self.b[1]-1:-1] # fill up bottom border
            pad_target[:,self.b[2]-1:None:-1] = pad_target[:,self.b[2]:2*self.b[2]] # fill up left border
            pad_target[:,-self.b[3]:] = pad_target[:,-self.b[3]-1:-2*self.b[3]-1:-1] # fill up right border

            # mirror weight
            pad_weight[self.b[0]:-self.b[1],self.b[2]:-self.b[3]] = weight 
            pad_weight[:self.b[0],self.b[2]:-self.b[3]] = weight[self.b[0]-1:None:-1]# fill up top border
            pad_weight[-self.b[1]:,self.b[2]:-self.b[3]] = weight[:-self.b[1]-1:-1] # fill up bottom border
            pad_weight[:,self.b[2]-1:None:-1] = pad_weight[:,self.b[2]:2*self.b[2]] # fill up left border
            pad_weight[:,-self.b[3]:] = pad_weight[:,-self.b[3]-1:-2*self.b[3]-1:-1] # fill up right border

            loc = [randint(0,16-1), randint(0,16-1)]
            image = pad_image[loc[0]:loc[0]+self.height, loc[1]:loc[1]+self.width]
            target = pad_target[loc[0]:loc[0]+self.height, loc[1]:loc[1]+self.width]
            weight = pad_weight[loc[0]:loc[0]+self.height, loc[1]:loc[1]+self.width]
        
        if random.random() < self.prob2:
            '''
            flipping
            '''

            image = np.flip(image,1)
            target = np.flip(target,1)
            weight = np.flip(weight,1)
        return image, target, weight
                
class ImdbData(Dataset):
    
    def __init__(self, config, X, y, W, transform=None):
        self.X = X
        self.y = y
        self.w = W
        self.height = config['general']['HEIGHT']
        self.width = config['general']['WIDTH']
        self.layers = config['general']['layers']
        self.transform = transform

    def __getitem__(self, index):
        img = np.transpose(self.X[index], (1,2,0)) 
        label = np.transpose(self.y[index],(1,2,0))
        weight = self.w[index]
        if self.transform is not None:
            img, label, weight = self.transform(img, label, weight)

        img = torch.from_numpy(img.copy()).float().permute(2,0,1)
        label = torch.from_numpy(label.copy()).long().permute(2,0,1)
        weight = torch.from_numpy(weight.copy()).float()
    
        return img, label, weight


    def __len__(self):
        return len(self.X)
                
    
with open( "./train.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
HEIGHT = config['general']['HEIGHT']
WIDTH = config['general']['WIDTH']
layers = config['general']['layers']
exp_dir_name = config['filepaths']['exp_dir_name']

model_path = config['filepaths']['model_path']
data_dir = config['filepaths']['processed_data_path']
param = config['param']

train_images, train_labels, train_wmaps, val_images, val_labels, val_wmaps = get_imdb_data(data_dir)

train_images2 = np.copy(np.expand_dims(train_images.reshape(-1,HEIGHT, WIDTH), axis = 1))
train_labels2 = np.copy(train_labels.reshape(-1, layers, HEIGHT, WIDTH))
train_wmaps2 = np.copy(train_wmaps.reshape(-1, HEIGHT, WIDTH))

val_images2 = np.copy(np.expand_dims(val_images.reshape(-1,HEIGHT, WIDTH), axis = 1))
val_labels2 = np.copy(val_labels.reshape(-1, layers, HEIGHT, WIDTH))
val_wmaps2 = np.copy(val_wmaps.reshape(-1, HEIGHT, WIDTH))

# combining train and validation since paper only did train
train_images3 = np.concatenate((train_images2, val_images2), axis=0)
train_labels3 = np.concatenate((train_labels2, val_labels2), axis=0)
train_wmaps3 = np.concatenate((train_wmaps2, val_wmaps2), axis=0)   

random_transform = RandomTransforms(config['general']['HEIGHT'], config['general']['WIDTH'], config['general']['layers'])

train_dataset = ImdbData(config, train_images3, train_labels3, train_wmaps3, transform = random_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=4)
device = torch.device("cuda")

relaynet_model = ReLayNet(param)
solver = Solver(device)
num_epochs = 60
solver.train(relaynet_model, train_loader, model_path=model_path, num_epochs=num_epochs, log_nth=1,  exp_dir_name=exp_dir_name)
