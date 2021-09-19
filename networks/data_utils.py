"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import h5py


class ImdbData(data.Dataset):
    def __init__(self, X, y, w=None):
        self.X = X
        self.y = y
        self.w = w

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        if self.w is not None:
            wmap = self.w[index]
            wmap = torch.from_numpy(wmap)
            return img, label, wmap
        else:
            return img, label

    def __len__(self):
        return len(self.y)


def get_imdb_data(data_dir):
    # TODO: Need to change later
    NumClass = 9

    # Load DATA
    
    with h5py.File(os.path.join(data_dir,'training_intermediate.hdf5'),'r') as hf: 
        train_images=hf['data'][()]
        train_labels=hf['lmap'][()]
        train_wmaps=hf['wmap'][()]

    with h5py.File(os.path.join(data_dir,'val_intermediate.hdf5'),'r') as hf: 
        val_images=hf['data'][()]
        val_labels=hf['lmap'][()]
        val_wmaps=hf['wmap'][()]
        

    return train_images, train_labels, train_wmaps, val_images, val_labels, val_wmaps
