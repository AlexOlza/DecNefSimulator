#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Currently, utility functions to generate initialization lattices
Created on Fri Aug 29 13:16:13 2025

@author: alexolza
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#%%

def bidirectional_reduction(dataset, latent=True, dim=2):
    # if latent==True, original dim: vae latent space (zdim=256 for example)
    # else: receives fMRI data generated from a VAE
    # fMRI--VAEenc-->256--PCAtransf-->2--PCAinv_transf-->256--VAEdec-->fMRI
    pca_pipe = Pipeline([('scaler', StandardScaler()),
                         ('pca', PCA(n_components=dim))])
    
    X = dataset.latents.numpy() if latent else dataset.reconstructions.numpy()
    y = dataset.labels.numpy().astype(int)
    pca_pipe = pca_pipe.fit(X)
    var = pca_pipe['pca'].explained_variance_ratio_ * 100
    title = f'Explained variance ratio: {var[0]:.2f}% + {var[1]:.2f}% ={var[0] + var[1]:.2f}% '
    print(title)
    dataset_transf = pca_pipe.transform(X)
    df = pd.DataFrame(np.hstack((dataset_transf, y.reshape(-1,1))), columns = [f'PC{i}' for i in range(1,dim+1)]+ ['Class'])
    df.Class = df.Class.astype(int).astype(str)
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    sns.scatterplot(data = df, x='PC1', y='PC2', hue = 'Class', ax=ax)
    fig.suptitle(title)
    return pca_pipe, df
def compute_latents_reconstructions(dataset, vae, device='cuda:1'):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    reconstructions = None
    latents = None
    labels = None

    vae.eval()
    vae.to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon, *_ = vae(x)
            latent, *_ = vae.encoder(x)
            reconstructions = recon.cpu() if reconstructions is None else torch.cat((reconstructions, recon.cpu()), dim=0)
            latents = latent.cpu() if latents is None else torch.cat((latents, latent.cpu()), dim=0)
            labels = np.array(y).ravel() if labels is None else np.hstack((labels, np.array(y).ravel()))

    # concatenate all batches into two big tensors
    # reconstructions = torch.cat(reconstructions, dim=0)
    # labels = torch.cat(labels, dim=0)
    labels = np.array(labels).ravel()
    print(reconstructions.shape, latents.shape, labels.shape)
    return reconstructions, latents, labels
class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, reconstructions, latents, labels):
        self.reconstructions = reconstructions
        self.latents = latents
        self.labels = torch.Tensor(labels).int()
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.reconstructions[idx], self.latents[idx], self.labels[idx]


def sample_gaussian(center, N, sigma):
    # Sample in continuous space
    pts = np.random.multivariate_normal(center, sigma, size=N)
    return pts  # ensure no duplicates

def make_init_z_lattice(N, z_dim, all_class_prototypes, all_class_prototypes_sigma, tgt_non_tgt, 
                        lattice_fname, z_grid_init_fname,
                        percent_tgt = 0.10, percent_non_tgt = 0.10,
                        sigma_gauss=0.5, margin = 0.25):
    tgt, non_tgt = tgt_non_tgt
    N_A = np.ceil(N/len(all_class_prototypes)).astype(int)
    
    # --- Sample near each prototype ---
    z_grid_init_= []
    prototype_class=0
    for prototype, sigma in zip(all_class_prototypes, all_class_prototypes_sigma):
        z_grid_init_A = sample_gaussian(prototype,
                                         N_A, 
                                         np.diag(sigma)
                                         )
        
        prot_idx = np.zeros((N_A,1)).astype(int) if prototype_class==0 else prototype_class*np.ones((N_A,1)).astype(int)
        print(z_grid_init_A.shape, prot_idx.shape)
        z_grid_init_A = np.hstack((z_grid_init_A, prot_idx))
        z_grid_init_ = np.vstack((z_grid_init_, z_grid_init_A)) if len(z_grid_init_)>0 else z_grid_init_A
        prototype_class += 1
    print(z_grid_init_.shape)
    np.save(z_grid_init_fname, z_grid_init_)
    return z_grid_init_

def load_dataset(dataset, transform=None, npz_file_path=None, train=True):
    if dataset=='FASHION':
        # Download and load the MNIST training data
        trainset = datasets.FashionMNIST('../data', download=True, train=train, transform=transform)
    elif dataset.startswith('synth_fMRI'):
        trainset = NPZDataset(npz_file_path, train=train)
    return trainset
#%%
class NPZDataset(Dataset):
    """
    CLASS MAPPING IN THE OFFICIAL FASHION DATASET:
        {'T-shirt/top': 0,
         'Trouser': 1,
         'Pullover': 2,
         'Dress': 3,
         'Coat': 4,
         'Sandal': 5,
         'Shirt': 6,
         'Sneaker': 7,
         'Bag': 8,
         'Ankle boot': 9}
    """
    def __init__(self, npz_file_path, train=True):
        """
        Parameters
        ----------
        npz_file_path : str
            DESCRIPTION.
        train : bool, optional
            Whether to use the train partition. The default is True.

        Returns
        -------
        torch-compatible dataset

        """
     
        npz_file = np.load(npz_file_path)
        try:
            self.original_img_indices = npz_file['original_img_indices'][npz_file['train_idx']==int(train)].ravel()
        except:
            print('No original indices in the npz file')
        self.targets = npz_file['y_int'][npz_file['train_idx']==int(train)].ravel()
        classes = npz_file['y'][npz_file['train_idx']==int(train)].ravel()
        data = torch.from_numpy(npz_file['X'][npz_file['train_idx']==int(train)])
        if torch.any(data.isnan(),dim=1).sum()>0:
            print(f'Warning: Dropping {torch.any(data.isnan(),dim=1).sum()} rows with NaN')
            data = data[~torch.any(data.isnan(),dim=1)]
        self.data = data
        class_map = pd.DataFrame(np.vstack((classes, self.targets))).T.drop_duplicates()
        self.targets = torch.from_numpy(self.targets).long()
        self.class_to_idx = {x:y for x,y in zip(class_map[0].values, class_map[1].values.astype(int))}
        self.classes = torch.from_numpy(np.sort(list(self.class_to_idx.values())))#np.unique(self.targets)       
        npz_file.close()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]
        # Convert NumPy arrays to PyTorch tensors
        sample = sample.float()
        label = label.long()
        return sample, label

