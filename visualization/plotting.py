#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Tue Mar 11 15:16:47 2025

@author: alexolza
"""

import torch
from torch.distributions.normal import Normal
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from umap.umap_ import UMAP
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from components.generators import get_data_predictions, get_classes_mean
import matplotlib as mpl

mpl.rcParams["font.family"] = "DejaVu Serif"   # or another installed system font you like
# Use Computer Modern for math
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.default"] = "bf"


# Global font size
mpl.rcParams["font.size"] = 14 # APPROPRIATE WITH FIGSIZE 8, 8

def compare_evolution_with_CI(dfs: dict, title: str, ylabel: str,
                              ax=None, fname=None, stat='mean', relative=True):
    if ax is None:  figure, ax = plt.subplots(figsize=(8, 6))
    for dfname, df in dfs.items():
        evolution_with_CI(df, None, # title=None to avoid titles in all axes
                          ylabel, ax, legend_label = dfname,  fname=None, stat=stat, relative=relative) # we save at the end
    ax.legend()
    plt.suptitle(title)
    if fname:
        plt.savefig(fname)
        
def evolution_with_CI(df: pd.DataFrame, title, ylabel,
                      ax=None, fname=None, legend_label='value',stat='mean', relative=True):
    df = df/df.values[0]
    df['time'] = df.index  # Add time index
    statfun = np.median if stat=='median' else np.mean
    # Melt the DataFrame so each row is (time, sample, value)
    df_long = df.melt(id_vars='time', var_name='sample', value_name=legend_label)

    if ax is None:  figure, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df_long, x='time', y=legend_label, label = legend_label,
                 estimator=statfun, errorbar=('ci', 95), ax= ax)
    if title: ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    
    if fname:
        plt.tight_layout()
        plt.savefig(fname)
    


def show_images_grid(images, class_num, ax, title, title_color='black', text=[], title_fontsize=26):
    grid = make_grid(images, nrow=class_num, normalize=True).permute(1,2,0).numpy()
    ax.imshow(grid)
    ax.set_title(title, color = title_color, fontsize=title_fontsize)
    plt.axis('off')
    return ax
def show_image(image, ax, title='', cmap='inferno'):
    try:
        ax.imshow(image.permute(1,2,0).numpy(), cmap=cmap)
        ax.set_title(title)
        plt.axis('off')
    except:
        print('Skipping show_image. Is your data tabular?')
    return ax

def traverse_two_latent_dimensions(model, input_sample, z_dist, n_samples=25, z_dim=16, dim_1=0, dim_2=1, title='plot', device='cuda',digit_size=28):
  

  percentiles = torch.linspace(1e-6, 0.9, n_samples)

  grid_x = z_dist.icdf(percentiles[:, None].repeat(1, z_dim))
  grid_y = z_dist.icdf(percentiles[:, None].repeat(1, z_dim))

  figure = np.zeros((digit_size * n_samples, digit_size * n_samples))

  z_sample_def = input_sample.clone().detach()
  target_size = torch.Size([digit_size, digit_size])
  # select two dimensions to vary (dim_1 and dim_2) and keep the rest fixed
  for yi in range(n_samples):
      for xi in range(n_samples):
          with torch.no_grad():
              z_sample = z_sample_def.clone().detach()
              z_sample[:, dim_1] = grid_x[xi, dim_1]
              z_sample[:, dim_2] = grid_y[yi, dim_2]
              x_decoded = model.decoder(z_sample.to(device),target_size).cpu()
              # print(x_decoded.shape)
          digit = x_decoded[0].reshape(digit_size, digit_size)
          figure[yi * digit_size: (yi + 1) * digit_size,
                 xi * digit_size: (xi + 1) * digit_size] = digit.numpy()

  fig, ax = plt.subplots(figsize=(15,15))
  ax.imshow(figure, cmap='Greys_r')
  plt.axis('off')
  plt.title(title)
  return fig

def obtain_latents(model, data_loader, class_names, num_samples, device):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
          if len(latents) > num_samples:
            break
          mu, _ = model.encoder(data.to(device))
          latents.append(mu.cpu())
          labels.append(label.cpu())

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return latents, labels

def visualize_latent_space(model, data_loader, named_prototypes, device, method='TSNE', num_samples=10000):
    model.eval()
    class_names = named_prototypes.keys()
    print(class_names)
    latents, labels = obtain_latents(model, data_loader, class_names, num_samples, device)
    assert method in ['TSNE', 'UMAP'], 'method should be TSNE or UMAP'
    if method == 'TSNE':
        tsne = TSNE(n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(latents)
        df = pd.DataFrame()
        df['TSNE-1'] = tsne_results[:, 0]
        df['TSNE-2'] = tsne_results[:, 1]
        df['Class'] = [list(class_names)[i] for i in labels.astype(int)]
        df = df.sort_values('Class')
        fig_data, ax_data = plt.subplots(1, figsize = (20, 10))
        sns.scatterplot(
                    x="TSNE-1", y="TSNE-2",
                    hue='Class',  # coloring
                    data=df,
                    alpha=1, ax=ax_data, palette = sns.color_palette("tab10")
                )
        ax_data.set_title(f'{latents.shape[-1]}-D VAE Latent Space with TSNE',fontsize= 22)  
    elif method == 'UMAP':
        reducer = UMAP()
        embedding = reducer.fit_transform(latents)
        df = pd.DataFrame()
        df['UMAP-1'] = embedding[:, 0]
        df['UMAP-2'] = embedding[:, 1]
        df['Class'] = [list(class_names)[i] for i in labels.astype(int)]
        df = df.sort_values('Class')
        fig_data, ax_data = plt.subplots(1, figsize = (20, 10))
        sns.scatterplot(
                    x="UMAP-1", y="UMAP-2",
                    hue='Class',  # coloring
                    data=df,
                    alpha=1, ax=ax_data, palette = sns.color_palette("tab10")
                )
        ax_data.set_title(f'{latents.shape[-1]}-D VAE Latent Space with UMAP ',fontsize= 22)
    return fig_data

def visual_eval_vae(vae, z_dim, train_loader, named_prototypes, device='cuda', reduction='UMAP'):
    xlim = 3
    vae.eval()
    vae_history = vae.history_to_df()
    fig_loss, ax = plt.subplots(1,2)
    bce = vae_history.train_BCE.plot(title = f'BCE loss (z_dim = {z_dim})', ax = ax[0])
    kl = vae_history.train_KL.plot(title = f'KL divergence (z_dim = {z_dim})',ax=ax[1])
    i=0
    rec, ax = plt.subplots(2, 3)
    try:
        for img, lbl in train_loader:
            while len(img.shape)<3:
                img = img.unsqueeze(-1)
            img = img[0][0] 
            ax[0][i].imshow(img, cmap='gray') 
            ax[1][i].imshow(vae(img.unsqueeze(0).unsqueeze(0).to(device))[0].detach().cpu()[0,0], cmap='gray')
            ax[0][i].set_title(lbl[0].item())
            
            i+=1
            if i>=3: break
        plt.axis('off')
        rec.suptitle(f'Reconstruction (z_dim={z_dim})')
    except:
        print('Skipping reconstructions. Is your data tabular?')
        rec=None
    
    latents_mean, latents_stdvar, labels = get_data_predictions(vae, train_loader)
    classes_mean = get_classes_mean(train_loader, labels, latents_mean, latents_stdvar)

    prot, axs = plt.subplots(1, len(named_prototypes), figsize=(4*len(named_prototypes),4))
    axss = axs.flatten()
    i=0

        
    for idx, label in enumerate(named_prototypes.keys()):
        latents_mean_target, latents_stddev_target = classes_mean[int(idx)]
        target_prototype = vae.decoder(torch.Tensor(latents_mean_target).to(device), vae.target_size)
        axss[i]=show_image(target_prototype[0].detach().cpu(),axss[i], title=f'{label}', cmap='inferno')
        axss[i].axis('off')
        i+=1
   
    if z_dim>2:
        latent_vis_umap = visualize_latent_space(vae, train_loader, named_prototypes,
                               device='cuda' if torch.cuda.is_available() else 'cpu',
                               method='UMAP', num_samples=10000)
        latent_vis_tsne = visualize_latent_space(vae, train_loader, named_prototypes,
                               device='cuda' if torch.cuda.is_available() else 'cpu',
                               method='TSNE', num_samples=10000)
    else:
        latent_vis_umap, ax = plt.subplots(figsize=(8,8))
        latents, labels = obtain_latents(vae, train_loader, named_prototypes.keys(), num_samples=10000, device=device)
        prototypes = np.vstack([prot[0].ravel() for idx, prot in vae.prototypes.items()
                                           ])
        print('prototypes.shape ', prototypes.shape)
        
        df = pd.DataFrame(latents, columns=['z_x', 'z_y'])
        df['Class'] = [list(named_prototypes.keys())[i] for i in labels.astype(int)]
        df = df.sample(frac=1)
        sns.scatterplot(
                    x="z_x", y="z_y",
                    hue='Class',  # coloring
                    data=df,
                    zorder=10,
                    alpha=0.8, ax=ax, palette = sns.color_palette("tab10")
                )
        ax.scatter(
            prototypes[:, 0],
            prototypes[:, 1],
            color="black",
            marker="X",
            s=30,       # bigger size
            zorder=15,
            alpha=1
        )
        # Annotate each prototype
        for label, prototype in named_prototypes.items():
            print(label)
            print(prototype)
            (x, y) = prototype[0][0].ravel()
            ax.text(
                x + 0.04, y,      # shift to the right
                label,
                fontsize=10,
                zorder=17,
                va="center",
                ha="left",
                color="black",
                weight="bold",
                label="_nolegend_"
            )
        ax.set_aspect("equal")
            
        ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.35),  # push outside
                ncol=4,
                # borderaxespad=0
                )
        
        ax.set_xlabel(r"$z_x$"); ax.set_ylabel(r"$z_y$")     
        ax.set_xlim([-xlim, xlim]); ax.set_ylim([-xlim, xlim])
        latent_vis_umap.tight_layout()
        latent_vis_tsne = None
    z_dist = Normal(torch.zeros(1, 2), torch.ones(1, 2))
    input_sample = torch.zeros(1, 2)
    if z_dim==2:
        try:
            latent_trav = traverse_two_latent_dimensions(vae, input_sample, z_dist, n_samples=20, dim_1=0, dim_2=1, z_dim=2, title='Traversing 2D latent space', device=device)
        except:
            latent_trav = None# plt.figure()
            print('Skipping latent space traversal. Is your data tabular?')
    else:
        latent_trav = None#plt.figure()
    return fig_loss, rec, prot, latent_vis_umap, latent_vis_tsne, latent_trav
