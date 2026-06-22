#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Functions to train a fMRI-to-image Multilayer Perception
(Figure 5)
Created on Thu Jun 18 14:49:34 2026

@author: alexolza
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

class SubsetToDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
 
    def __getitem__(self, index):
        return self.subset[index]
 
    def __len__(self):
        return len(self.subset)
    
class VoxelImageDataset(Dataset):
    def __init__(self, images, voxels):
        """
        images: Tensor [N, 1, 28, 28]
        voxels: Tensor [N, V]
        """
        self.images = images.float()
        self.voxels = voxels.float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.voxels[idx]




class VoxelToImage(torch.nn.Module):
    def __init__(self, n_voxels):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_voxels, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 28 * 28),
            torch.nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, voxels):
        img = self.decoder(voxels)
        return img.view(-1, 1, 28, 28)

def train(model, loader, device, epochs=20):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, voxels in loader:
            images = images.to(device)
            voxels = voxels.to(device)

            pred = model(voxels)
            loss = F.mse_loss(pred, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:03d} | MSE: {total_loss/len(loader):.4f}")


@torch.no_grad()
def reconstruct(model, voxels, device):
    """
    voxels: Tensor [V] or [1, V]
    """
    model.eval()
    voxels = voxels.to(device)

    if voxels.ndim == 1:
        voxels = voxels.unsqueeze(0)

    pred = model(voxels)
    return pred.cpu()
