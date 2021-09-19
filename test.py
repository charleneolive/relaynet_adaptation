import os
import glob
import scipy
import scipy.io
import cv2
import h5py
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from preprocess.data_prep_utils import retinaFlatten as rf
from preprocess.data_prep_utils import octSpectralisReader as rd
from preprocess.data_prep_utils.misc import build_mask
from networks.data_utils import ImdbData

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable 

torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "Region above the retina (RaR)", "rgb_values": [128, 0, 0]},
    {"id": 1, "name": "ILM: Inner limiting membrane", "rgb_values": [0, 128, 0]},
    {"id": 2, "name": "NFL-IPL: Nerve fiber ending to Inner plexiform layer", "rgb_values": [128, 128, 0]},
    {"id": 3, "name": "INL: Inner Nuclear layer", "rgb_values": [0, 0, 128]},
    {"id": 4, "name": "OPL: Outer plexiform layer", "rgb_values": [128, 0, 128]},
    {"id": 5, "name": "ONL-ISM: Outer Nuclear layer to Inner segment myeloid", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "ISE: Inner segment ellipsoid", "rgb_values": [128, 128, 128]},
    {"id": 7, "name": "OS-RPE: Outer segment to Retinal pigment epithelium", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Region below RPE (RbR)", "rgb_values": [192, 0, 0]}];


def prepare_dataset(raw_data_path, label_path):
    # for count,dataset in enumerate(image_datasets):
    [header, BScanHeader, slo, BScans] = rd.octSpectralisReader(raw_data_path)
    header['angle'] = 0

    mat = scipy.io.loadmat(label_path)
    annotations=mat['bd_pts']

    background=np.ones(BScans.shape)
    layers_map = np.zeros((annotations.shape[2], BScans.shape[0], BScans.shape[1], BScans.shape[2]))

    for scan in range(annotations.shape[1]):
        layers_map[:,:,:,scan] = build_mask(annotations[:,scan,:],height,width)
    layers_map[layers_map.shape[0]-1]=background-np.sum(layers_map[:-1,:,:,:],0)
    return layers_map, BScans

def label_img_to_rgb(label_img):

    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)
    

with open( "./test.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

data_dir = config['filepaths']['data_dir']
dataset_name = config['filepaths']['dataset_name']
save_path = config['filepaths']['save_path']
test_dataset = os.path.join(data_dir,'processed', dataset_name, config['filepaths']['test_dataset'])
height = config['general']['height']
width = config['general']['width']
layers = config['general']['layers']

relaynet_model =  torch.load(config['filepaths']['model_path'])
relaynet_model.eval()

all_test_cases = []
with open(test_dataset,'r') as reader:
    for idx, line in enumerate(reader.readlines()):
        all_test_cases.append(line.strip('\n'))
predicted_segmentations = np.zeros((len(all_test_cases), 49, layers, height, width))

for case in all_test_cases:  
    patient_name = os.path.splitext(case)[0]
    raw_data_path = os.path.join(data_dir,'raw', dataset_name, patient_name+'.vol')
    label_path=os.path.join(data_dir,'labels', dataset_name,patient_name+'_label.mat')
    
    layers_map, BScans = prepare_dataset(raw_data_path, label_path)
    BScans2 = np.expand_dims(np.transpose(BScans, (2, 0, 1)), axis = 1)
    layers_map2 = np.transpose(layers_map, (3, 0, 1, 2))

    test_dataset = ImdbData(BScans2, layers_map2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    for idx2, (img, label) in enumerate(test_loader):
        with torch.no_grad():
            out = relaynet_model(Variable(img.cuda()))

        out = F.softmax(out,dim=1)
        predicted_segmentations[idx, idx2] = np.squeeze(out.data.cpu().numpy())
        max_val, rgb_img = torch.max(out,1)
        rgb_img = rgb_img.data.cpu().numpy()
        rgb_img = label_img_to_rgb(rgb_img[0])

        Path(os.path.join(save_path, patient_name)).mkdir(parents=True, exist_ok = True)
        cv2.imwrite(os.path.join(save_path, patient_name, "scan"+str(idx2)+".png"), rgb_img)
            
with h5py.File(os.path.join(save_path,'predictions'+'.hdf5'), 'w') as hf:
    hf.create_dataset('predictions', data=predicted_segmentations)
hf.close()