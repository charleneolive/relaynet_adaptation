from random import shuffle
import numpy as np
import torch.nn.functional as F
import torch
import pathlib
import pandas as pd
from torch.autograd import Variable
from networks.net_api.losses import CombinedLoss
from torch.optim import lr_scheduler
import os
from tqdm import tqdm
def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    for i in range(num_class):
        GT = y_true == (i)
        Pred = y_pred == (i)
        inter = np.sum(np.multiply(GT, Pred)) + 0.0001
        union = np.sum(GT) + np.sum(Pred) + 0.0001
        t = 2 * inter / union
        avg_dice = avg_dice + (t / num_class)
    return avg_dice

def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    for i in range(num_class):
        GT = y_true == (i)
        Pred = y_pred == (i)
        inter = np.sum(np.multiply(GT, Pred)) + 0.0001
        union = np.sum(GT) + np.sum(Pred) + 0.0001
        t = 2 * inter / union
        avg_dice = avg_dice + (t / num_class)
    return avg_dice

class Solver(object):
    # global optimiser parameters
    default_optim_args = {"lr": 0.1,
                          "momentum" : 0.9,
                          "weight_decay": 0.0001}
    gamma = 0.1
    step_size = 30
    NumClass = 10 # TO CHANGE

    def __init__(self, device, optim=torch.optim.SGD, optim_args={}):
        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = CombinedLoss(device)
        self.device = device

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []

    def train(self, model, train_loader, model_path, num_epochs=10, log_nth=5, exp_dir_name='exp_default'):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        # learning rate schedular
        scheduler = lr_scheduler.StepLR(optim, step_size=self.step_size,
                                        gamma=self.gamma)  # decay LR by a factor of 0.1 every 30 epochs

        
        iter_per_epoch = 1
        # iter_per_epoch = len(train_loader)

        model.to(self.device)

        print('START TRAIN.')
        curr_iter = 0
        
        per_epoch_train_acc = []

        for epoch in range(num_epochs):
            scheduler.step()
            self._reset_histories()
            model.train()
            iteration = 0
            
            batch = tqdm(enumerate(train_loader), total=len(train_loader))
            
            for i_batch, sample_batched in batch:
                X = Variable(sample_batched[0], requires_grad=True)
                y = Variable(sample_batched[1])
                w = Variable(sample_batched[2])

                if model.is_cuda:
                    X, y, w = X.cuda(), y.cuda(), w.cuda()
                optim.zero_grad()
                output = model(X)
                loss = self.loss_func(output, y, w)
                loss.backward()
                optim.step()
                _,batch_output =torch.max(F.softmax(output, dim=1), dim=1)
                _, y = torch.max(y, dim=1)
                avg_dice = per_class_dice(batch_output, y, self.NumClass)
                self.train_loss_history.append(loss.detach().item())
                
                self.train_acc_history.append(avg_dice)
            per_epoch_train_acc.append(np.sum(np.asarray(self.train_acc_history))/len(train_loader))
            
            print('[Epoch : {} / {}]: {:.2f}'.format(epoch, num_epochs, avg_dice.item()))
            
            full_save_path = os.path.join(model_path, exp_dir_name)
            pathlib.Path(full_save_path).mkdir(parents=True, exist_ok=True)
            model.save(os.path.join(full_save_path, 'relaynet_epoch'+ str(epoch + 1) + '.model'))

        d = {'train_history': per_epoch_train_acc}
        df = pd.DataFrame(data=d)
        df.to_csv(os.path.join(full_save_path, 'accuracy_history.csv'))
        print('FINISH.')
        
# class Solver(object):
#     # global optimiser parameters
#     default_optim_args = {"lr": 0.1,
#                           "momentum" : 0.9,
#                           "weight_decay": 0.0001}
#     gamma = 0.1
#     step_size = 30
#     NumClass = 9 # TO CHANGE

#     def __init__(self, optim=torch.optim.SGD, optim_args={},
#                  loss_func=CombinedLoss()):
#         optim_args_merged = self.default_optim_args.copy()
#         optim_args_merged.update(optim_args)
#         self.optim_args = optim_args_merged
#         self.optim = optim
#         self.loss_func = loss_func

#         self._reset_histories()

#     def _reset_histories(self):
#         """
#         Resets train and val histories for the accuracy and the loss.
#         """
#         self.train_loss_history = []
#         self.train_acc_history = []
#         self.val_acc_history = []
#         self.val_loss_history = []

#     def train(self, model, train_loader, val_loader, model_path, num_epochs=10, log_nth=5, exp_dir_name='exp_default'):
#         """
#         Train a given model with the provided data.

#         Inputs:
#         - model: model object initialized from a torch.nn.Module
#         - train_loader: train data in torch.utils.data.DataLoader
#         - val_loader: val data in torch.utils.data.DataLoader
#         - num_epochs: total number of training epochs
#         - log_nth: log training accuracy and loss every nth iteration
#         """
#         optim = self.optim(model.parameters(), **self.optim_args)
#         # learning rate schedular
#         scheduler = lr_scheduler.StepLR(optim, step_size=self.step_size,
#                                         gamma=self.gamma)  # decay LR by a factor of 0.1 every 30 epochs

        
#         iter_per_epoch = 1
#         # iter_per_epoch = len(train_loader)

#         if torch.cuda.is_available():
#             model.cuda()

#         print('START TRAIN.')
#         curr_iter = 0
        
#         per_epoch_train_acc = []
#         per_epoch_val_acc = []

#         for epoch in range(num_epochs):
#             scheduler.step()
#             self._reset_histories()
#             model.train()
#             iteration = 0
            
#             batch = tqdm(enumerate(train_loader), total=len(train_loader))
            
#             for i_batch, sample_batched in batch:
#                 X = Variable(sample_batched[0])
#                 y = Variable(sample_batched[1])
#                 w = Variable(sample_batched[2])

#                 if model.is_cuda:
#                     X, y, w = X.cuda(), y.cuda(), w.cuda()
#                 optim.zero_grad()
#                 output = model(X)
#                 loss = self.loss_func(output, y, w)
#                 loss.backward()
#                 optim.step()
#                 _,batch_output =torch.max(F.softmax(output, dim=1), dim=1)
#                 _, y = torch.max(y, dim=1)
#                 avg_dice = per_class_dice(batch_output, y, self.NumClass)
#                 self.train_loss_history.append(loss.detach().item())
                
#                 self.train_acc_history.append(avg_dice)
#             per_epoch_train_acc.append(np.sum(np.asarray(self.train_acc_history))/len(train_loader))
            
#             model.eval()
#             batch = tqdm(enumerate(val_loader), total=len(val_loader))
#             for i_batch, sample_batched in batch:
#                 with torch.no_grad():
#                     X = Variable(sample_batched[0])
#                     y = Variable(sample_batched[1])
#                     w = Variable(sample_batched[2])
#                     if model.is_cuda:
#                         X, y, w = X.cuda(), y.cuda(), w.cuda()
                
#                 _, val_output = torch.max(F.softmax(model(X), dim= 1),dim=1)
#                 _, y = torch.max(y, dim=1)
#                 avg_dice = per_class_dice(val_output, y, self.NumClass)
            
#                 self.val_acc_history.append(avg_dice.item())
#             per_epoch_val_acc.append(np.sum(np.asarray(self.val_acc_history))/len(val_loader))
#             print('[Epoch : {} / {}]: {:.2f}'.format(epoch, num_epochs, avg_dice.item()))
            
#             full_save_path = os.path.join(model_path, exp_dir_name)
#             pathlib.Path(full_save_path).mkdir(parents=True, exist_ok=True)
#             model.save(os.path.join(full_save_path, 'relaynet_epoch'+ str(epoch + 1) + '.model'))

#         d = {'train_history': per_epoch_train_acc, 'validation history': per_epoch_val_acc}
#         df = pd.DataFrame(data=d)
#         df.to_csv(os.path.join(full_save_path, 'accuracy_history.csv'))
#         print('FINISH.')