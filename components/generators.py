#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementing Latent Variable Generative models used in 
DecNefLab: A Modular and Interpretable Simulation Framework for Decoded Neurofeedback
(Olza et al.)
https://arxiv.org/abs/2511.14555

Created on Fri Feb  7 14:56:30 2025

@author: alexolza

inspired by: 
    https://hackernoon.com/how-to-sample-from-latent-space-with-variational-autoencoder
    https://github.com/qbxlvnf11/conditional-GAN/blob/main/conditional-GAN-generating-fashion-mnist.ipynb
    https://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and-code.html
    
"""
from torch import nn
import torch
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np
import pandas as pd
def kl_divergence_loss(z_dist):
    return kl_divergence(z_dist,
                         Normal(torch.zeros_like(z_dist.mean),
                                torch.ones_like(z_dist.stddev))
                         ).sum(-1).sum()




class Encoder(nn.Module):
    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        """
        Encoder class for a convolutional VAE, designed for image data.
        It can work as a standalone 2-block CNN too.
        It is only implemented with 2 convolutional blocks.
        
        Parameters
        ----------
        im_chan : int, optional
            DESCRIPTION: Image channels (grayscale: 1, color - RGB: 3).
        output_chan : int, optional
            DESCRIPTION: Output size. This will be the VAE's latent space dimension.
        hidden_dim : int, optional
            DESCRIPTION: Output size of the hidden CNN block.
        """
        super(Encoder, self).__init__()
        self.z_dim = output_chan

        self.encoder = nn.Sequential(
            self.init_conv_block(im_chan, hidden_dim),
            self.init_conv_block(hidden_dim, hidden_dim * 2),
            # double output_chan for mean and std with [output_chan] size
            
            self.init_conv_block(hidden_dim * 2, output_chan * 2, final_layer=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def init_conv_block(self, input_channels: int, output_channels: int,
                        kernel_size: int=4, stride: int=2, padding: int=0,
                        final_layer: bool=False):
        layers = [
            nn.Conv2d(input_channels, output_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          stride=stride)
        ]
        if not final_layer:
            layers += [
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def forward(self, image):
        encoder_pred = self.encoder(image)
        encoding = encoder_pred.view(len(encoder_pred), -1)
        mean = encoding[:, :self.z_dim]
        logvar = encoding[:, self.z_dim:]
        # encoding output representing standard deviation is interpreted as
        # the logarithm of the variance associated with the normal distribution
        # take the exponent to convert it to standard deviation
        return mean, torch.exp(logvar*0.5)
    

class Decoder(nn.Module):
    def __init__(self, z_dim: int=32, im_chan: int=1, hidden_dim: int=64):
        """
        Decoder class for a 2-block convolutional VAE, designed for image data.
        Parameters
        ----------
        z_dim : int, optional
            DESCRIPTION: Latent space dim of the VAE,
                         the dimension entering the Decoder
        im_chan : int, optional
            DESCRIPTION: Image channels (grayscale: 1, color - RGB: 3).
        hidden_dim : int, optional
            DESCRIPTION: Output size of the hidden CNN block.
        """
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.decoder = nn.Sequential(
            self.init_conv_block(z_dim, hidden_dim * 4),
            self.init_conv_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.init_conv_block(hidden_dim * 2, hidden_dim),
            # nn.AdaptiveAvgPool2d((1, 1)),
            self.init_conv_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def init_conv_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        layers = [
            nn.ConvTranspose2d(input_channels, output_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding)
        ]
        if not final_layer:
            layers += [
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            ]
        else:
            layers += [nn.Sigmoid()]
        return nn.Sequential(*layers)

    def forward(self, z, target_size):
        # Ensure the input latent vector z is correctly reshaped for the decoder
        x = z.view(-1, self.z_dim, 1, 1)
        # Pass the reshaped input through the decoder network
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class VAE(nn.Module):
    def __init__(self, z_dim: int=32, im_chan: int=1, tabular: bool=False,
                 n_features: int=None, device: str='cuda'):
      """
        Variational AutoEncoder, accepts 2 versions:
            - CNN-based (images): Uses classes Encoder and Decoder above
            - MLP-based (tabular data): Uses classes Encoder_1D and Decoder_1D
        Parameters
        ----------
        z_dim : int, optional
            DESCRIPTION: Latent space dimension
        im_chan : int, optional
            DESCRIPTION: Image channels. Only used if CNN-based
        tabular : bool, optional
            DESCRIPTION: If true: MLP VAE. If False: CNN VAE
        n_features : int, required for tabular data
            DESCRIPTION: If Tabular, number of data features. Else, unused. 
        device : str, optional
            DESCRIPTION: Whether to work in CPU or GPU
      """
      super(VAE, self).__init__()
      self.z_dim = z_dim
      self.device = device
      self.tabular = tabular
      self.n_features = n_features
      
      if self.tabular: 
          assert self.n_features>1, 'To use with tabular data, provide n_features at init!'
          self.reconstruction_loss = nn.MSELoss(reduction='sum')
          self.encoder = Encoder_1D(n_features, z_dim)
          self.decoder = Decoder_1D(n_features, z_dim)
      else:
          self.encoder = Encoder(im_chan, z_dim)
          self.decoder = Decoder(z_dim, im_chan)
          self.reconstruction_loss = nn.BCELoss(reduction='sum')
      self.optimizer = torch.optim.Adam(self.parameters())
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'extra_attributes': {
                'target_size': self.target_size,
                'history': self.history,
                'prototypes': self.prototypes
            }
        }, path)
        
    def load(self, path, strict=True):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        extras = checkpoint.get('extra_attributes', {})
        for key, value in extras.items():
            setattr(self, key, value)
    
    
    def forward(self, X):
        """
        Forward pass in training.
        In the first forward pass, sets self.target_size dinamically. 
        Parameters
        ----------
        X : Input tensor. Goes through the Encoder.

        Returns
        -------
        decoding : Tensor. DESCRIPTION: Reconstruction of X
        z_dist : Distribution.
        DESCRIPTION: Normal distribution,
                     with mean and variance given by the encoding of X           
        """
        if not self.tabular:
            target_size = X.shape[-2:]  # image (height, width)
        else:
            target_size = X.shape[-1]  # n_features
        if not hasattr(self, 'target_size'): 
            print(f'Setting target size to {target_size} (X.shape={X.shape}')
            self.target_size = target_size
        
        z_dist = Normal(*self.encoder(X))
        # sample from distribution with reparametrization trick
        z = z_dist.rsample()
        decoding = self.decoder(z, target_size)
        return decoding, z_dist

    def training_step(self, batch, beta):
        X = batch.to(self.device)
        recon_X, encoding = self(X)
        bce = self.reconstruction_loss(recon_X, X)#/(torch.Tensor([np.prod(list(model.target_size))]).to(device))
        kl = kl_divergence_loss(encoding)
        loss = bce+ beta*kl 
        return loss, bce, kl
    
    def fit(self, train_loader, epochs,  annealing_epochs = 0, max_beta = 1, verbose=0):
        self.verbose = verbose
        history = []
        if annealing_epochs>0: betas = np.linspace(0.1,0.5,annealing_epochs)
        else: betas = max_beta*np.ones(epochs)
        for epoch in range(epochs):
            epoch_loss, epoch_bce, epoch_kl = [], [], []
            beta = betas[epoch] if epoch< len(betas) else max_beta
            self.train()
            for batch, step in train_loader:
                loss, bce, kl = self.training_step(batch, beta)
                epoch_loss.append(loss); epoch_bce.append(bce); epoch_kl.append(kl)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()       
            # self.epoch_end(epoch, result) # only for printing (when verbose)
            history.append([torch.stack(epoch_loss).mean().item(),
                            torch.stack(epoch_bce).mean().item(),
                            torch.stack(epoch_kl).mean().item()])
            self.history = history
            if self.verbose: print(f'Epoch {epoch}/{epochs}: loss {loss}')
    def history_to_df(self):
        return pd.DataFrame(self.history,
                            columns = ['train_loss','train_BCE','train_KL'])

    def compute_prototypes(self, data_loader):
        """
        For each class of data in a labelled dataset of categorical data,
        compute the prototype of that class in the VAE's latent space.
        LATENT PROTOTYPE (DEFINITION):
            Mean representation, in the latent space, of a set of data
        Detailed explanation:
            DecNefLab: A Modular and Interpretable Simulation Framework for Decoded Neurofeedback
            (Olza et al.)
            https://arxiv.org/abs/2511.14555
        Parameters
        ----------
        data_loader : Torch Dataloader (data and labels)

        """
        self.eval()
        latents_mean = []
        latents_std = []
        labels = []
        with torch.no_grad():
            for i, (data, label) in enumerate(data_loader):
              mu, std = self.encoder(data.to(self.device))
              latents_mean.append(mu.cpu())
              latents_std.append(std.cpu())
              labels.append(label.cpu())
        latents_mean = torch.cat(latents_mean, dim=0)
        latents_std = torch.cat(latents_std, dim=0)
        labels = torch.cat(labels, dim=0)
        classes_mean = {}
        for class_name in np.unique(labels):
          latents_mean_class = latents_mean[labels==class_name]
          latents_mean_class = latents_mean_class.mean(dim=0, keepdims=True).detach().numpy()

          latents_std_class = latents_std[labels==class_name]
          latents_std_class = latents_std_class.mean(dim=0, keepdims=True).detach().numpy()

          classes_mean[class_name] = [latents_mean_class, latents_std_class]
        
        self.prototypes = classes_mean  

def nan_check(t, name):
    if torch.isnan(t).any():
        print("NaN in", name)
        raise ValueError()

class Encoder_1D(Encoder):   
    def __init__(self, n_features, z_dim=32, hidden_dim=1024):
        super(Encoder_1D, self).__init__()
        self.z_dim = z_dim
        self.input_dim = n_features
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim // 2, self.z_dim * 2)  # outputs mean and logvar
        )
  
    def forward(self, x):
        # x: (batch_size, input_dim)
        nan_check(x, 'x enc')
        encoding = self.encoder(x)
        mean = encoding[:, :self.z_dim]
        logvar = encoding[:, self.z_dim:]
        nan_check(mean, 'mean enc')
        nan_check(logvar, 'logvar enc')
        return mean, torch.exp(0.5 * logvar)

class Decoder_1D(nn.Module):
    def __init__(self, n_features_out, z_dim=32, hidden_dim=1024):
        super(Decoder_1D, self).__init__()
        self.z_dim = z_dim
        self.output_dim = n_features_out
        self.hidden_dim = hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_dim, self.output_dim),
            # nn.Sigmoid()  # optional depending on your data
        )

    def forward(self, z, target_size): # target_size arg only for compatibility with convolutional version
        return self.decoder(z)

def get_data_predictions(model, data_loader, device='cuda'):
    model.eval()
    latents_mean = []
    latents_std = []
    labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
          mu, std = model.encoder(data.to(device))
          latents_mean.append(mu.cpu())
          latents_std.append(std.cpu())
          labels.append(label.cpu())
    latents_mean = torch.cat(latents_mean, dim=0)
    latents_std = torch.cat(latents_std, dim=0)
    labels = torch.cat(labels, dim=0)
    return latents_mean, latents_std, labels

def get_classes_mean(train_loader, labels, latents_mean, latents_std,):
  classes_mean = {}
  for class_name in np.unique(labels):
    latents_mean_class = latents_mean[labels==class_name]
    latents_mean_class = latents_mean_class.mean(dim=0, keepdims=True)

    latents_std_class = latents_std[labels==class_name]
    latents_std_class = latents_std_class.mean(dim=0, keepdims=True)

    classes_mean[class_name] = [latents_mean_class, latents_std_class]
  return classes_mean