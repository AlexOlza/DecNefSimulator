#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:12:20 2025

@author: alexolza
"""
import sys
sys.path.append('..')
import torch
import os
import regex as re
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
###########################
from utils import load_dataset
from components.classifiers import CNNClassification, ElasticNetLinearClassification, BinaryDataLoader
from config_files.traditional_decnef_n_instances import traditional_decnef_n_instances_parser
#%%
"""
Configuration variables
"""
global_random_seed = 42
config = traditional_decnef_n_instances_parser()

device = config.device
outpath = f'../EXPERIMENTS/{config.EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{config.EXP_NAME}/weights/'
classifier_type =  'ELASTICNET' if config.dataset.startswith('synth_fMRI') else 'CNN'
classifier_epochs = 10
classifier_batch_size = 16
tgt_non_tgt = [config.target_class_idx, config.non_target_class_idx]
#%%
for p in [outpath, modelpath]:
    if not os.path.exists(p):
        os.makedirs(p)
#%%
dataset = config.dataset
transform = transforms.Compose([transforms.ToTensor()]) if dataset=='FASHION' else None
trainset = load_dataset(dataset, transform, config.npz_file_path, train= True)
testset = load_dataset(dataset, transform, config.npz_file_path, train= False)

if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}

class_name_dict_reverse = {v: k for k, v in class_name_dict.items()}    
class_names, class_numbers = np.array([[k,v] for k,v in class_name_dict.items()]).T
class_numbers = class_numbers.astype(int)

img_size = trainset[0][0].shape[-1]
#%%
combo_names = [list(class_names)[i] for i in tgt_non_tgt]
discr_str = f'{combo_names[0]} vs {combo_names[1]}'
clean_discr_str = re.sub('[^a-zA-Z0-9]','', discr_str)
print(clean_discr_str)

classifier_name = f'{classifier_type}_{clean_discr_str}__BS{classifier_batch_size}_E{classifier_epochs}'
classifier_fname = os.path.join(modelpath, classifier_name+'.pt')

if classifier_type=='CNN':
	# torch.Size([1, 28, 28] is hardcoded because I'm using FASHION-MNIST and this is the shape of images. To use with other image datasets, this needs to be altered (programatically, through the img_size variable)
    classifier = CNNClassification(torch.Size([1, 28, 28]), tgt_non_tgt, device, name=discr_str) 
    learning_rate = 1e-3
else:
    classifier = ElasticNetLinearClassification(img_size, tgt_non_tgt, device=device, name=discr_str)
    learning_rate = 1e-4
print(classifier)

if not os.path.exists(classifier_fname):
    print('CLASSIFIER TRAINING: ',discr_str)
    tl = BinaryDataLoader(trainset, tgt_non_tgt, batch_size=16)
    testl = BinaryDataLoader(testset, tgt_non_tgt, batch_size=16) 
    classifier.evaluate(testl)
    classifier.fit( epochs=classifier_epochs, learning_rate=learning_rate, train_loader=tl, val_loader = testl)
    classifier.save(classifier_fname)
    print('Performance: ')
    print(classifier.history_to_df())
    print(f'Saved {classifier_fname}, exiting')
else:
    print(f'Found {classifier_fname}, exiting')
