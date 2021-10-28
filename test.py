import os
import glob
import scipy
import scipy.io
import cv2
import sys
import h5py
import yaml
import sklearn
import warnings
import json
import seaborn as sns
import pandas as pd
import sklearn.metrics
import torch.utils.data as data
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from preprocess.data_prep_utils import octSpectralisReader as rd
from preprocess.data_prep_utils.misc import build_mask

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable 

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

with open( "./test.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

filepaths = config['filepaths']
test_dataset_list = os.path.join(filepaths['project_dir'],'data','processed', filepaths['dataset_name'], filepaths['test_dataset'])
dimensions = config['general']
layers = config['layers']

relaynet_model =  torch.load(filepaths['model_path'])
relaynet_model.eval()

all_test_cases = []
with open(test_dataset_list,'r') as reader:
    for idx, line in enumerate(reader.readlines()):
        all_test_cases.append(line.strip('\n'))
        
class EvalPatient:
    def __init__(self, patient, dimensions, dataset_name, layers, save_path):
        self.patient = patient
        self.dimensions = dimensions
        self.height = dimensions["height"]
        self.width = dimensions["width"]
        self.dataset_name = dataset_name
        self.layers = layers
        self.removed_layers = {'MIAMI_HC':4} # ONL layer
        
        self.metrics = None
        self.overall_metrics = {k:[] for k in ["IOU","precision","recall","f1"]}
         
        self.save_path = save_path
        self.viz_path = os.path.join(self.save_path, 'visualisations_match', patient)
        self.viz_path_overlay = os.path.join(self.save_path, 'visualisations_match_overlay', patient)
        
        
        self.patient_eval_logs = []
        self.patient_eval_stats = {k: {g:[] for g in ['precision','recall','IOU','f1']} for k in layers[dataset_name].keys()}
        
        Path(self.viz_path).mkdir(exist_ok=True, parents=True)
        Path(self.viz_path_overlay).mkdir(exist_ok=True, parents=True)
        
    def saveViz(self, num, pred_rgb, gt_rgb, actual_img):
        
        cv2.imwrite('{}/Pred_{}_Slice_{}.png'.format(self.viz_path, self.patient, num), pred_rgb)
        cv2.imwrite('{}/Ground_{}_Slice_{}.png'.format(self.viz_path, self.patient, num), gt_rgb)
        cv2.imwrite('{}/Actual_{}_Slice_{}.png'.format(self.viz_path, self.patient, num), np.uint8(actual_img*255))
    
    def saveStack(self,pred_rgb_stack, gt_rgb_stack):
        with h5py.File('{}/Stack_{}.hdf5'.format(self.viz_path,self.patient), 'w') as hf:
            hf.create_dataset("pred", data=pred_rgb_stack)
            hf.create_dataset("gt", data=gt_rgb_stack)
        hf.close()        
        
    def layerMetrics(self, binary_img, gt_img, layer_name):
        '''
        layer based metrics after stitching up image
        params: 
        binary_img: image of predicted segmentation for 1 layer
        gt_img: image of ground truth segmentation for 1 layer
        layer_name: layer name
        
        output:
        patient_eval_stats: dictionary of all the patient-level metrics arranged by metric and layer name
        metrics: all the metrics by metric type
        '''
        intersection = np.logical_and(gt_img, binary_img)
        union = np.logical_or(gt_img, binary_img)
        
        tn, fp, fn, tp  = sklearn.metrics.confusion_matrix((gt_img).ravel()>0, (binary_img).ravel()).ravel()
        IOU = np.sum(intersection) / np.sum(union)

        def fxn():
            warnings.warn("runtime", RuntimeWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
            precision = tp/(tp+fp) # will lead to runtime warnings
            
        recall = tp/(tp+fn)
        f1 = (2*tp)/((2*tp)+fp+fn)
        
        metric_names = ['precision', 'recall', 'IOU', 'f1']
        metrics = [precision, recall, IOU, f1]
        
        for name, m in zip(metric_names, metrics):
            self.patient_eval_stats[layer_name][name].append(m)
            self.metrics[name].append(m)
            

    def getStats(self, patient_labels, patient_predictions, patient_images):
        
        # layer based matrics
        pred_rgb_stack = np.zeros((*patient_images.shape, 3))
        gt_rgb_stack = np.zeros((*patient_images.shape, 3))
        
        for num in range(patient_images.shape[0]):
            bscan_label = patient_labels[num]
            bscan_prediction = patient_predictions[num]
            bscan_image = patient_images[num]
            bscan_prediction[9] = bscan_prediction[0]+bscan_prediction[9]
            bscan_label_max = np.argmax(bscan_label, axis=0)
            bscan_prediction_max = np.argmax(bscan_prediction[1:], axis=0)

            self.metrics = {k:[] for k in ["IOU","precision","recall","f1"]}
            # for visualisation purpose
            pred_rgb = np.zeros((dimensions["height"],dimensions["width"], 3))
            gt_rgb = np.zeros((dimensions["height"],dimensions["width"], 3))
            
            for idx2, (layer_name, color) in enumerate(self.layers[self.dataset_name].items()):
                binary_img = (bscan_prediction_max == int(idx2))
                gt_img = (bscan_label_max == int(idx2))
                self.layerMetrics(binary_img, gt_img, layer_name)
                pred_rgb[bscan_prediction_max==idx2] = color
                gt_rgb[bscan_label_max==idx2] = color
            

            mean_precision, mean_recall, mean_iou, mean_f1 = tuple(map(lambda x: np.mean(self.metrics[x]),  \
                                                                 ["precision", "recall", "IOU", "f1"]))

            pred_rgb_stack[idx] = pred_rgb
            gt_rgb_stack[idx] = gt_rgb


            # evaluation per b-scan
            self.patient_eval_logs.append([self.patient, num, mean_iou, mean_precision, mean_recall, mean_f1])
            self.saveViz(num, pred_rgb, gt_rgb, bscan_image)
        self.saveStack(pred_rgb_stack, gt_rgb_stack)
        

def plot_result(save_path, eval_stats, eval_logs, layers, dataset_name):
    eval_logs = [stat for patient in eval_logs for stat in patient]

    with open('{}/evaluation_stats.json'.format(save_path),'w') as fp:
        json.dump(eval_stats, fp)

    df = pd.DataFrame(eval_logs, columns = ['patient', 'scan', 'mean iou', 'mean precision', 'mean recall','mean f1'])
    df['patient_no'] = df['patient'].astype(str).str.split('_').str[0].str.replace('0','').str[2:].astype(int)
    df.sort_values(by=['patient_no'])
    df.to_csv('{}/evaluation_logs.csv'.format(save_path), index=False)
                
    plt.figure(figsize=(20,10))
    ax =  sns.boxplot(data = df, y = "mean iou", x = "patient_no", color = "skyblue", width=0.5)
    [x.set_linewidth(4) for x in ax.spines.values()]
    ax.set_xlabel("Patient Number",fontsize=30, labelpad=10)
    ax.set_ylabel("Mean IOU",fontsize=30, labelpad=10)
    plt.xticks(fontsize= 25)
    plt.yticks(fontsize= 25)
    plt.savefig("{}/iou.png".format(save_path), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # layers
    all_stats = []

    # reordering to plot properly
    new_eval_stats = {k: {g:[] for g in ['precision','recall','IOU','f1']} for k in layers[dataset_name].keys()}
    # reformat the eval_stats dictionary
    for layer in new_eval_stats.keys():
        for metric in ['precision','recall','IOU','f1']:
            new_eval_stats[layer][metric] = [num for p in eval_stats.keys() for num in eval_stats[p][layer][metric]]

    for metric in ['IOU', 'precision', 'recall', 'f1']:
        # append by layer
        all_stats.append([new_eval_stats[k][metric] for k in new_eval_stats.keys()])

    all_stats_name = ['IOU', 'Precision','Recall','F1 score']

    for stat, name in zip(all_stats, all_stats_name): # plot by metric type
        plt.figure(figsize=(20,10))
        # plot box plot by layer
        ax = sns.boxplot(data = [d for d in stat], color = "skyblue", width=0.5)
#             sns.despine(offset=10, trim=True)
        ax.set_xticklabels(new_eval_stats.keys())
        [x.set_linewidth(4) for x in ax.spines.values()]
#             ax.axes.set_title("Graph of mean {} across ground truth layers".format(name),fontsize=40)
        ax.set_xlabel("Layers",fontsize=30, labelpad=10)
        ax.set_ylabel("{}".format(name),fontsize=30, labelpad=10)
        plt.xticks(fontsize= 25)
        plt.yticks(fontsize= 25)
        plt.savefig("{}/{}_by_layer.png".format(save_path, name), dpi=300, bbox_inches='tight', transparent=True)

        plt.close()
        
        
def prepare_dataset(raw_data_path, label_path, dimensions):
    # for count,dataset in enumerate(image_datasets):
    [header, BScanHeader, slo, BScans] = rd.octSpectralisReader(raw_data_path)
    header['angle'] = 0

    mat = scipy.io.loadmat(label_path)
    annotations=mat['bd_pts']

    background=np.ones(BScans.shape)
    layers_map = np.zeros((annotations.shape[2], BScans.shape[0], BScans.shape[1], BScans.shape[2]))

    for scan in range(annotations.shape[1]):
        layers_map[:,:,:,scan] = build_mask(annotations[:,scan,:],dimensions['height'],dimensions['width'])
    layers_map[layers_map.shape[0]-1]=background-np.sum(layers_map[:-1,:,:,:],0)
    return layers_map, BScans


class ImdbData(data.Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        img = self.X[index]
        img = torch.from_numpy(img)
        return img

    def __len__(self):
        return len(self.X)

def getPredictions(BScans):
    stitched_stack = np.zeros((dimensions['bscans'], 
                                    dimensions['layers'], dimensions['height'], dimensions['width']))
    test_dataset = ImdbData(BScans)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    for idx, (img) in enumerate(test_loader):
        with torch.no_grad():
            out = relaynet_model(Variable(img.cuda()))
        out = F.softmax(out,dim=1)
        
        stitched_stack[idx] = np.squeeze(out.data.cpu().numpy())
        
    return stitched_stack


all_predictions = []
full_save_path = os.path.join(filepaths['project_dir'], filepaths['save_path'], filepaths['dataset_name'])
Path(full_save_path).mkdir(exist_ok=True, parents=True)

eval_logs = []
for case in all_test_cases:  
    patient = os.path.splitext(case)[0]
    
    eval_patient = EvalPatient(patient, dimensions, filepaths['dataset_name'], layers, full_save_path)
    
    
    raw_data_path = os.path.join(filepaths['project_dir'],'data','raw',filepaths['dataset_name'], patient+'.vol')
    label_path=os.path.join(filepaths['project_dir'],'data','labels', filepaths['dataset_name'],patient+'_label.mat')
    
    
    layers_map, BScans = prepare_dataset(raw_data_path, label_path, dimensions)
    BScans2 = np.expand_dims(np.transpose(BScans, (2, 0, 1)), axis = 1)
    stitched_stack = getPredictions(BScans2)
    layers_map2 = np.transpose(layers_map, (3, 0, 1, 2))
    BScans2 = np.squeeze(BScans2, axis=1)
    
    eval_patient.getStats(layers_map2, stitched_stack, BScans2)
    
    eval_logs.append(eval_patient.patient_eval_logs)
    eval_stats[patient] = eval_patient.patient_eval_stats
            
    all_predictions.append(stitched_stack)
    
plot_result(full_save_path, eval_stats, eval_logs, layers, filepaths['dataset_name'])
with h5py.File(os.path.join(filepaths['predictions'],'predictions'+'.hdf5'), 'w') as hf:
    hf.create_dataset('pred', data=all_predictions)
hf.close()