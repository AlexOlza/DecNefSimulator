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
from matplotlib import pyplot as plt
from umap.umap_ import UMAP
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib as mpl
from matplotlib import gridspec

mpl.rcParams["font.family"] = "DejaVu Serif"
# Use Computer Modern for math
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.default"] = "bf"


# Global font size
mpl.rcParams["font.size"] = 14 # APPROPRIATE WITH FIGSIZE 8, 8


@torch.no_grad()
def heatmap_mean_X_over_time(X,
                             ALL_TRAJS,
                             trajectory_names,
                             latent_prototype,
                             cname_dict,
                             title=None, 
                             save=False,
                             fname = None,
                             p_mean = None,
                             offset=0, # to discard the warmup rounds
                             ext='png',
                             N_A=None,
                             figsize=(8,6)):
    if save: assert len(fname)>0, 'save required but no fname provided!' 
    cname_dict = {'T-shirt/Top': 0,
     'Trouser': 1,
     'Pullover': 2,
     'Dress': 3,
     'Coat': 4,
     'Sandal': 5,
     'Shirt': 6,
     'Sneaker': 7,
     'Bag': 8,
     'Ankleboot': 9} # Override it because of capitalization
    
    ALL_X = X.to_numpy() # shape (501, 1000)
    # p0_mean = ALL_X[offset, :].mean()
    print(f'p_mean = {p_mean}')
 
    X_groupbyR = []
    traj_cols = []
    traj2init = {}
    
    for i, (init, idx) in enumerate(trajectory_names.groupby("init").groups.items()):
        if idx.max() < ALL_X.shape[1]:
            # average trajectory
            X_groupbyR.append(ALL_X[:, idx].mean(axis=1))
            # name for the column
            col = f"tr{i}"
            traj_cols.append(col)
            # record mapping from new column name -> init
            traj2init[col] = init
    
    # make dataframe
    DF = pd.DataFrame(np.array(X_groupbyR).T, columns=traj_cols)

    print(DF.shape)
    
    if offset>0: DF = DF.iloc[offset:,:]
    
    # reshape to long-form
    DF["Time"] = DF.index
    df_long = DF.melt(id_vars="Time", var_name="traj", value_name="Probability")
    df_long["init"] = df_long["traj"].map(traj2init)
    df_long["init_bin"] = (df_long["init"] // 10) * 10
    print(df_long.shape)
    print(df_long["init_bin"].unique())
    
    # Reverse cname_dict: map number → class name
    num2class = {v: k for k, v in cname_dict.items()}
    
    # Map init_bin to class name
    # If your init_bin is 0,10,20,..., then divide by 10 and modulo 10 to get 0-9
    df_long["class_name"] = df_long["init_bin"].div(10).astype(int) % 10
    df_long["class_name"] = df_long["class_name"].map(num2class)
    print(df_long.class_name.unique())
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_ylim([0,1])
    sns.lineplot(
        data=df_long,
        x="Time",
        y="Probability",
        hue="class_name",   # color by bin (10 consecutive inits share color)
        units="traj",       
        estimator=None,
        alpha=0.4,
        linewidth=0.9,
        legend=False,
        palette='tab10',
        ax=ax,
        dashes=False        # required, otherwise seaborn sometimes skips drawing
    )
    
    # Step 2. Mean line per init (opaque)
    sns.lineplot(
        data=df_long,
        x="Time",
        y="Probability",
        hue="class_name",
        estimator="mean",
        palette='tab10',
        # ci=None,
        linewidth=2.5,
        alpha=1.0,
        legend=True,
        ax=ax,
        dashes=False
    )
    # ax = plt.gca()
    
    if p_mean is not None:
        ax.axhline(p_mean, linestyle='--', color='black', alpha=1)
        ax.annotate(f'Mean $p_{0} = {p_mean:.2f}$',(250, p_mean),
                   bbox = dict(facecolor='white', edgecolor='black', 
                               boxstyle='round,pad=0.6', alpha=0.6)
                   )
    else:
        ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
        
    plt.xticks(range(0, 501, 100));# plt.xticklabels([i for i in range(0, 501, 100)])
    # Adjust figure to avoid clipping
    # plt.subplots_adjust(bottom=0.27)  # leave room for legend
    # plt.subplots_adjust(right=0.9)    
    # Create and store legend artist
    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title=None
    )
    
    # Save including the legend
    fig.savefig(
        f'{fname}_lineplot_grouped.{ext}',
        bbox_extra_artists=[leg],
        bbox_inches='tight'
    )
    
@torch.no_grad()
def plot_probability_map_grid_(probability_map, coordinates, pca_pipe,
                               generator, z_dim, classifier,
                               target_class_idx, non_target_class_idx, class_name_dict, clean_discr_str,
                               n_samples, space_radius,
                               accuracy='', ext='png', maxmin=False, fname = 'pmap_pca_{clean_discr_str}_{n_samples}samples',
                               seed=42):   
    # 4) PLOT PROBABILITY MAP  
    xmin, ymin, xmax, ymax = coordinates[:,0].min(), coordinates[:,1].min(), coordinates[:,0].max(), coordinates[:,1].max()
    accuracy = classifier.history_to_df().val_acc.values[-1]
    dx = xmax - xmin
    dy = ymax - ymin
    vmin = 0 if maxmin else probability_map.min()
    vmax = 1 if maxmin else probability_map.max()
    idx = 0 if target_class_idx==min(classifier.classes) else 1
    figname = f'{fname}_maxmin.{ext}' if maxmin else f'{fname}.{ext}'
    if dy < dx:
        mid = 0.5 * (ymin + ymax)
        ymin = mid - dx / 2
        ymax = mid + dx / 2


    fig, ax = plt.subplots(1,1, figsize = (8, 8))
    im = ax.imshow(probability_map,
                   vmin = vmin, vmax = vmax,
                   extent = space_radius,
                   origin='lower', cmap='viridis', #aspect = 'equal'
                   ) 
    ax.set_xlabel(r'$PC_1$'+ '\n' + '(PCA visualization of $z$ coordinates)', labelpad=1)
    ax.set_ylabel(r'$PC_2$', labelpad=0.1)


    for cname, c in class_name_dict.items():
        print(c)
        print(generator.prototypes[c][0].shape)
        x, y = pca_pipe.transform(generator.prototypes[c][0]).ravel()
        ax.plot(x, y, marker = 'X', color = 'red', markeredgecolor='white',  markersize=18, zorder=12)
        if c==target_class_idx:
            fontweight="bold"
            fontsize=14
            if cname=='TSHIRTTOP': cname = 'T-shirt/top'
            tgt_name=cname.capitalize()
            text = f'{cname}\n(Target)'
        elif c==non_target_class_idx:
            text = f"{cname.capitalize()}\n(Alternative)"
            fontweight="bold"
            fontsize=14
            non_tgt_name=cname.capitalize()
        else:
            text = cname.capitalize()
            fontweight="normal"
            fontsize=12
        ax.annotate(text, (x, y), 
                    color='black', fontsize=fontsize, fontweight=fontweight,
                    # ha="center", va="center",
                    bbox = dict(facecolor='white', edgecolor='black', 
                                boxstyle='round,pad=0.3', alpha=0.6)
                    )

    ax.set_title(f'{tgt_name} vs. {non_tgt_name} - Accuracy: {accuracy:.2f}')
    cbar = fig.colorbar(im, ax = ax, fraction=0.04, pad = 0.04)
    cbar.set_label('Probability')
    fig.savefig(figname)

@torch.no_grad()
def plot_probability_map_grid(probability_map, coordinates, pca_pipe,
                                 generator, z_dim, classifier,
                                 target_class_idx, non_target_class_idx, class_name_dict, clean_discr_str,
                                 n_samples, space_radius,
                                 accuracy='', ext='png', maxmin=False, fname = 'pmap_{clean_discr_str}_{n_samples}samples',
                                 seed=42):
    np.random.seed(seed) 
    if z_dim==2: plot_probability_map_grid_2d(probability_map, coordinates, generator, z_dim, classifier,
                                     target_class_idx, non_target_class_idx, class_name_dict, clean_discr_str,
                                     n_samples, space_radius,
                                     accuracy, ext, maxmin, fname,
                                     seed)
    else:
        plot_probability_map_grid_(probability_map, coordinates, pca_pipe,
                                   generator, z_dim, classifier,
                                   target_class_idx, non_target_class_idx, class_name_dict, clean_discr_str,
                                   n_samples, space_radius,
                                   accuracy, ext, maxmin, fname,
                                   seed)

@torch.no_grad()
def plot_probability_map_grid_2d(probability_map, coordinates, generator, z_dim, classifier,
                                 target_class_idx, non_target_class_idx, class_name_dict, clean_discr_str,
                                 n_samples, space_radius,
                                 accuracy='', ext='png', maxmin=False, fname = 'pmap_{clean_discr_str}_{n_samples}samples',
                                 seed=42):
    xmin, xmax, ymin, ymax = space_radius
    vmin = 0 if maxmin else probability_map.min()
    vmax = 1 if maxmin else probability_map.max()
    idx = 0 if target_class_idx==min(classifier.classes) else 1
    print(maxmin, vmin, vmax)
    figname = f'{fname}_maxmin.{ext}' if maxmin else f'{fname}.{ext}'
    fig = plt.figure(figsize=(10, 10))
    
    # Define an 8x8 grid
    gs = gridspec.GridSpec(6,6 , figure=fig, wspace=0.9, hspace=0.65)
    
    # --- First column (8 stacked small axes) ---
    col_axes = []
    for i in range(6):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        col_axes.append(ax)
    
    # --- Bottom row (8 small axes) ---
    row_axes = []
    for j in range(1,6):
        ax = fig.add_subplot(gs[5, j])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        row_axes.append(ax)
    
    # --- Big axis spanning rows 0–6 and cols 1–7 ---
    h, w = probability_map.shape
    big_ax = fig.add_subplot(gs[0:5, 1:6])
    big_ax.set_aspect("equal")
    big_ax.set_xlabel(r'$z_{x}$', labelpad=1)
    big_ax.set_ylabel(r'$z_{y}$', labelpad=-4)
    im = big_ax.imshow(
        probability_map,
        extent=space_radius,
        origin="lower",
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        aspect=(w/h),
        # alpha=0.8
    )
    # Overlay prototypes
    prototypes = {}
    tgt_name, non_tgt_name = "", ""
    for cname, c in class_name_dict.items():
        if len(np.array(generator.prototypes[c]).flatten())==2*generator.z_dim: # contains mean and variance
            prototypes[c]  = generator.prototypes[c][0][0]
        else:
            prototypes[c] = generator.prototypes[c]
        x, y = prototypes[c]
             
        if c == target_class_idx:
            big_ax.plot(x, y, marker="X", color="red", markersize=18, zorder=12)
            big_ax.annotate(f"{cname}\n (Target)", (x+0.05, y-0.3), color="black",
                            fontsize=14,
                            fontweight="bold",
                            zorder=10,
                            # ha="center",
                            # va="center",
                            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.6)
                            )

            reconstructed_prototype = generator.decoder(torch.tensor(prototypes[c], device = generator.device),generator.target_size).cpu().detach()

            if len(reconstructed_prototype.shape)<3:
                reconstructed_prototype = reconstructed_prototype.unflatten(0, classifier.image_shape).unsqueeze(0)

            p = torch.nn.Softmax(dim=0)(classifier(reconstructed_prototype.to(classifier.device)).flatten())[idx]
   
            col_axes[0].imshow(reconstructed_prototype[0][0], cmap='inferno')
            tgt_name = cname
            col_axes[0].set_title(f'Target\n $p={p:.2f}$')
        elif c == non_target_class_idx:
            big_ax.plot(x, y, marker="X", color="red", markersize=10, zorder=12)
            big_ax.annotate(f"{cname}\n (Non-target)", (x, y-0.4), color="black", 
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.6)
            )

            reconstructed_prototype = generator.decoder(torch.tensor(prototypes[c], device = generator.device),generator.target_size).cpu().detach() 

            if len(reconstructed_prototype.shape)<3:
                reconstructed_prototype = reconstructed_prototype.unflatten(0, classifier.image_shape).unsqueeze(0)

            p = torch.nn.Softmax(dim=0)(classifier(reconstructed_prototype.to(classifier.device)).flatten())[idx]

            non_tgt_name = cname
            row_axes[-1].imshow(reconstructed_prototype[0][0], cmap='inferno')
            row_axes[-1].set_title(f'Non-target\n $p={p:.2f}$')
        else:
            big_ax.plot(x, y, marker="X", color="red", markersize=6, zorder=12)
            big_ax.annotate(cname, (x, y-0.1), color="black", 
            fontsize=12, zorder=10,
            # fontweight="bold",
            # ha="center",
            # va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.6)
            )
    # Add colorbar for the big axis
    cbar = fig.colorbar(im, ax=big_ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")

    # Select other coordinate points to reconstruct
    i=1
    remaining_axes = np.concatenate((col_axes[1:], row_axes[:-1])).ravel()
    points = np.random.uniform(max(xmin, ymin), min(xmax, ymax), size=(len(remaining_axes), 2))
    
    for point, ax in zip(points, remaining_axes):
        
        reconstruction = generator.decoder(torch.tensor(point, device = generator.device),generator.target_size).cpu().detach() 

        if len(reconstruction.shape)<3:
            reconstruction = reconstruction.unflatten(0, classifier.image_shape).unsqueeze(0)

        p = torch.nn.Softmax(dim=0)(classifier(reconstruction.to(classifier.device)).flatten())[idx]

        big_ax.annotate(f'({i})', point, color="red",
        fontsize=10,
        # fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.3", alpha=0.7)
        )
        ax.imshow(reconstruction[0][0], cmap='gray')
        ax.set_title(f'$({i})$ $p={p:.2f}$')
        i+=1
    if accuracy: big_ax.set_title(f'{tgt_name} vs. {non_tgt_name} - Accuracy: {accuracy}')
    plt.savefig(figname)
    plt.show()
