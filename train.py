import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.autograd import Variable
from networks.data_utils import get_imdb_data
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn.functional as F
from networks.relay_net import ReLayNet
from networks.solver import Solver

class RandomTransforms(object):
    def __init__(self, height, width, layers, prob1=0.5, prob2=0.5, prob3 = 0.5, h_size= 490, w_size= 60):
        self.height = height
        self.width = width
        self.layers = layers
        self.prob = prob1
        self.prob2 = prob2
        self.prob3 = prob3
        self.translate_prob = 0.5
        self.h_size = h_size
        self.w_size = w_size

    def __call__(self, image, target, weight):
        if random.random() < self.prob:

            image = np.flip(image,1)
            target = np.flip(target,1)
            weight = np.flip(weight,1)
            return image, target, weight
        
        if random.random()<self.prob2:

            y,x,c = image.shape
            startx = x//2 - self.w_size//2
            starty = y//2 - self.h_size//2   
            image = image[starty:starty+self.h_size, startx:startx+self.w_size, :].copy()
            target = target[starty:starty+self.h_size, startx:startx+self.w_size, :].copy()
            weight = weight[starty:starty+self.h_size, startx:startx+self.w_size].copy()
            image = np.resize(image, (self.height,self.width,1))
            target = np.resize(target, (self.height,self.width,self.layers))
            weight = np.resize(weight, (self.height,self.width))
            return image, target, weight
            
        if random.random()<self.prob3:

            if random.random()<self.translate_prob:

                pad_image = np.zeros(image.shape)
                pad_target = np.zeros(target.shape)
                pad_weight = np.zeros(weight.shape)
                y,x,c = image.shape
                startx = x//2 - self.w_size//2
                image = image[:, startx:startx+self.w_size, :].copy()
                pad_image[:,:image.shape[1],:] = image
                target = target[:, startx:startx+self.w_size, :].copy()
                pad_target[:,:image.shape[1],:] = target
                weight = weight[:, startx:startx+self.w_size].copy()
                weight[:,:image.shape[1]] = weight
                
                return pad_image, pad_target, pad_weight
            else:

                pad_image = np.zeros(image.shape)
                pad_target = np.zeros(target.shape)
                pad_weight = np.zeros(weight.shape)
                y,x,c = image.shape
                starty = y//2 - self.h_size//2 
                image = image[starty:starty+self.h_size, :, :].copy()
                pad_image[:image.shape[0],:,:] = image
                target = target[starty:starty+self.h_size, :, :].copy()
                pad_target[:image.shape[0],:,:] = target
                weight = weight[starty:starty+self.h_size, :].copy()
                pad_weight[:weight.shape[0],:] = weight
                
                return pad_image, pad_target, pad_weight
        return image, target, weight
                
class ImdbData(Dataset):
    
    def __init__(self, config, X, y, W, transform):
        self.X = X
        self.y = y
        self.w = W
        self.height = config['general']['HEIGHT']
        self.width = config['general']['WIDTH']
        self.layers = config['general']['layers']
        self.transform = transform(self.height, self.width, self.layers)

    def __getitem__(self, index):
        img = np.transpose(self.X[index], (1,2,0)) 
        label = np.transpose(self.y[index],(1,2,0))
        weight = self.w[index]
        if self.transform:

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

train_dataset = ImdbData(config, train_images2, train_labels2, train_wmaps2, transform = RandomTransforms)
val_dataset = ImdbData(config, val_images2, val_labels2, val_wmaps2, transform = RandomTransforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

relaynet_model = ReLayNet(param)
solver = Solver(optim_args={"lr": 1e-2})
        
solver.train(relaynet_model, train_loader, val_loader, model_path, log_nth=1, num_epochs=20, exp_dir_name=exp_dir_name)