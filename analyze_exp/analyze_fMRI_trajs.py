#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze results of DecNef simulations from
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Wed Dec 10 17:13:44 2025

@author: alexolza
"""

import matplotlib.cm as cm
# matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import os
import sys
sys.path.append('../')
import re
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib as mpl
###########################################
from utils import load_dataset, ReconstructionDataset, bidirectional_reduction
from visualization.plotting import visual_eval_vae
from components.generators import VAE
from components.discriminators import CNNClassification, ElasticNetLinearClassification
from visualization.plotting import heatmap_mean_X_over_time
from analysis.utils import trajectory_properties_as_df, get_probabilities
############################################

##########################################################################
#%%
"""
##############################################
CONFIGURATION VARIABLES
##############################################
"""
mpl.rcParams["font.family"] = "DejaVu Serif"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.default"] = "bf"
ext = 'pdf'
EXP_NAME  = sys.argv[1]
dataset = 'synth_fMRI_FASHION' if 'synth' in EXP_NAME else 'FASHION'

target_class_idx = int(eval(sys.argv[2])) 
non_target_class_idx = int(eval(sys.argv[3])) 
subj = int(eval(sys.argv[4])) 
decnef_iters =  500
n_trajs = 100
linv = 5
z_dim = 2 if (dataset=='FASHION' or 'z2' in EXP_NAME) else 256
lambda_ = 1/linv 
generator_epochs = 25 if dataset=='FASHION' else 25#300
space_radius=2.5 if dataset=='FASHION' else 3.5
n_samples=75
device='cuda:0'
generator_batch_size=64
discriminator_type = 'CNN' if dataset=='FASHION' else 'ELASTICNET'
generator_name = 'VAE'
discriminator_epochs = 10
discriminator_batch_size = 16
tgt_non_tgt = [target_class_idx, non_target_class_idx]
tgt, non_tgt= tgt_non_tgt
seed=7
outpath = f'../EXPERIMENTS/{EXP_NAME}/output/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/weights/'
repeat = 5 if subj==8 else 8
npz_file_paths = {'FASHION':'',
                  'synth_fMRI_FASHION':f'../data/fMRIsynth/subj0{subj}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri.npz',
                  'synth_fMRI_COCO':''}

npz_file_path = npz_file_paths[dataset]

transform = transforms.Compose([transforms.ToTensor()]) if dataset=='FASHION' else None
trainset = load_dataset(dataset, transform, npz_file_path=npz_file_path, train=True)
testset = load_dataset(dataset, transform, npz_file_path=npz_file_path, train=False)
train_loader = DataLoader(trainset, batch_size=64)
test_loader = DataLoader(testset, batch_size=64)
# testset = datasets.FashionMNIST('../data', download=True, train=False, transforms=PILToTensor())
if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
class_name_dict =  {'TSHIRTTOP': 0,
  'TROUSER': 1,
  'PULLOVER': 2,
  'DRESS': 3,
  'COAT': 4,
  'SANDAL': 5,
  'SHIRT': 6,
  'SNEAKER': 7,
  'BAG': 8,
  'ANKLEBOOT': 9}
class_name_dict_reverse = {v: k for k, v in class_name_dict.items()} 

combo_names = [class_name_dict_reverse[i] for i in tgt_non_tgt]
clean_discr_str = re.sub('[^a-zA-Z0-9]','', f'{combo_names[0]} vs {combo_names[1]}')
figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/{clean_discr_str}/nfb_eval/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/figures/{clean_discr_str}/nfb_eval/'

if not os.path.exists(figpath): os.makedirs(figpath)

discriminator_name = f'{discriminator_type}_{clean_discr_str}__BS{discriminator_batch_size}_E{discriminator_epochs}'
discriminator_fname = os.path.join(modelpath, discriminator_name+'.pt')
generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

keys = {'With memory (CNN)': ['MNDAVMem', 0],
        'With memory (Random)': ['MNDAVMem', 1],
        'MNDAV (CNN)': ['MNDAV', 0],
        'MNDAV (Random)': ['MNDAV', 1]
        }

seed_list = [] 
for i in range(100):
    seeds = [ s for s in range((i+1)*42,
    (i+1)*42 + 10)]
    seed_list.append(seeds)

z_grid_init_fname = os.path.join(modelpath, f'z_grid_init_{clean_discr_str}.npy')
z_grid_init = np.load(z_grid_init_fname)
original_gaussian = z_grid_init[:,-1].astype(int)
seed_list = np.array(seed_list).flatten()
map_traj_orig = {s: o for s, o in zip(seed_list, original_gaussian.repeat(10))}

#%%
"""
LOAD MODELS
"""
img_size = 14386 # TODO: assign programatically# trainset[0][0].shape[-1]
tabular= True if 'fMRI' in dataset else None
n_features = img_size if 'fMRI' in dataset else None
if dataset=='FASHION':
    discriminator = CNNClassification(torch.Size([1, 28, 28]), tgt_non_tgt, device='cuda:0') 
else:
    discriminator = ElasticNetLinearClassification(img_size, tgt_non_tgt, device='cuda:0')

discriminator.load(discriminator_fname)
discriminator.eval()
vae = VAE(z_dim=z_dim, tabular=tabular, n_features=img_size, device= device).to(device)
vae.target_size = 2 if dataset=='FASHION' else 256
vae.load(generator_fname+'.pt')
vae.eval()
latent_prototype = vae.prototypes[target_class_idx][0] # [1] is the variance and [0] is the mu
alt_latent_prototype = vae.prototypes[non_target_class_idx][0]
prototype = vae.decoder(torch.Tensor(latent_prototype).to(vae.device),
                                            vae.target_size).detach()
all_class_prototypes = {class_name_dict_reverse[k]: v for k,v in vae.prototypes.items()}

#%%
"""
##############################################
LOADING RESULTS
##############################################
"""
probability_dfs, sigma_dfs, trajectory_matrices = {},{},{}
names_dfs = {}

print('processing results...')
for UR, URname in tqdm(zip(['MNDAV', 'MNDAVMem'], ['MNDAV', 'With memory'])):
    for IGDIS, IGDIS_label in zip([0,1], ['CNN', 'Random']):
            trajectory_dir = os.path.join(outpath,f'TRAJS_{generator_name}_{discriminator_name}', f'linv{linv}',f'UR{UR}',f'IGDIS{IGDIS}')
            
            trajectory_names = glob(f'{dataset}_TRAJ*_z0*_{generator_name}_{discriminator_name}_UR{UR}_IGDIS{IGDIS}_linv{linv}.npz',
                                    root_dir=trajectory_dir)
            
            trajectory_paths = [os.path.join(trajectory_dir, f'{trajectory_name}')
                                for trajectory_name in trajectory_names]
            print(f'N={len(trajectory_names)}; {UR}, {IGDIS_label}')
            label = f'{URname} ({IGDIS_label})'    
            probability_dfs[label], sigma_dfs[label],\
                    trajectory_matrices[label], names_dfs[label] =trajectory_properties_as_df(trajectory_paths,
                                                                                             decnef_iters, 
                                                                                                 prototype,
                                                                                                 latent_prototype)      

for key in trajectory_matrices.keys():
    try:
        names_dfs[key]["init"] = names_dfs[key]["traj_name"].str.extract(r"z0*(\d+)_").astype(int)
        names_dfs[key]["rep"] = names_dfs[key]["traj_name"].str.extract(r"TRAJ*(\d+)_").astype(int)
        names_dfs[key]["linv"] = names_dfs[key]["traj_name"].str.extract(r"linv*(\d+).npz").astype(int)
    except:
        print('pass: ', key)
all_dfs = {'p': probability_dfs, 
           'sigma': sigma_dfs, 
          }
#%%
"""
##############################################
GENERATING PLOTS
##############################################
"""
for dl, mode in zip([train_loader, test_loader],['train','test']):
    fig_loss, rec, prot, latent_vis_umap, latent_vis_tsne, latent_trav = visual_eval_vae(vae, z_dim, dl, all_class_prototypes, device='cuda')

    
    fnames = [f'{figpath}/{figname}_{generator_name}_{mode}.{ext}'
              for figname in ['loss', 'rec', 'prototypes',
                              'latent_vis_umap', 'latent_vis_tsne', 'latent_trav']]
    
    for fig, fname in zip([fig_loss, rec, prot,
                           latent_vis_umap, latent_vis_tsne, latent_trav],
                          fnames):
        if fig is not None:
            try:
                fig.tight_layout()
                fig.savefig(fname)
                print('saved: ', fname)
            except:
                print(f'Skipped {fname}')

#%%
"""
PROBABILITY MAP EQUIVALENT
"""
# 1) PROJECT LATENT CLASS PROTOTYPES TO 2D USING PCA
latent_prototypes = None
prototypes = None
labels = []
vae.eval()
vae.to(device)
with torch.no_grad():
    for y, x in vae.prototypes.items():
        proto = torch.Tensor(x[0]).to(device)
        recon, *_ = vae.decoder(proto, vae.target_size)
        prototypes = recon.cpu() if prototypes is None else torch.cat((prototypes, recon.cpu()), dim=0)
        latent_prototypes = torch.Tensor(x[0]) if latent_prototypes is None else torch.cat((latent_prototypes, torch.Tensor(x[0])), dim=0)
        labels.append(y)
labels = np.array(labels)

prototype_dataset = ReconstructionDataset(prototypes, latent_prototypes, labels)
pca_pipe_proto, pca_df_proto = bidirectional_reduction(prototype_dataset, latent=True, dim=2)
pca_inv_proto = pca_pipe_proto.inverse_transform(pca_df_proto[['PC1', 'PC2']])
#%%
# 2) FIX BOUNDARIES FOR THE PLOT
xmin, xmax = pca_df_proto.PC1.min(), pca_df_proto.PC1.max()
ymin, ymax = pca_df_proto.PC2.min(), pca_df_proto.PC2.max()

# 3) SAMPLE UNIFORMLY FROM THE GRID, compute probability map
x_vals = np.linspace(xmin, xmax, n_samples)
y_vals = np.linspace(ymin, ymax, n_samples)
generated_samples = []
coordinates = []
probability_map = np.empty((n_samples, n_samples))
idx = 0
with torch.no_grad():
    for i, y in tqdm(enumerate(y_vals), total = len(y_vals)):
        for j, x in enumerate(x_vals):
            z = torch.Tensor(pca_pipe_proto.inverse_transform(np.array([x,y]).reshape(1, -1))).to(vae.device)
            generated_sample = vae.decoder(z, target_size=vae.target_size)
            p = torch.nn.Softmax(dim=0)(
                discriminator(
                    generated_sample.to(discriminator.device)
                    ).flatten()
                )[idx]
            generated_samples.append(generated_sample)
            probability_map[i, j] = p.cpu().numpy()
            coordinates.append([x, y])
#%%
# 4) PLOT PROBABILITY MAP  
accuracy = discriminator.history_to_df().val_acc.values[-1]
dx = xmax - xmin
dy = ymax - ymin

if dy < dx:
    mid = 0.5 * (ymin + ymax)
    ymin = mid - dx / 2
    ymax = mid + dx / 2


fig, ax = plt.subplots(1,1, figsize = (8, 8))
im = ax.imshow(probability_map,
               vmin=0, vmax=1,
               extent = [xmin-5, xmax+5, ymin-1, ymax+1], # TODO: no magic numbers!
               origin='lower', cmap='viridis', #aspect = 'equal'
               ) 
ax.set_xlabel(r'$PC_1$'+ '\n' + '(PCA visualization of $z$ coordinates)', labelpad=1)
ax.set_ylabel(r'$PC_2$', labelpad=0.1)


for cname, c in class_name_dict.items():
    x, y = pca_df_proto.iloc[c][['PC1', 'PC2']]
    ax.plot(x, y, marker = 'X', color = 'red', markeredgecolor='white',  markersize=18, zorder=12)
    if c==tgt:
        fontweight="bold"
        fontsize=14
        if cname=='TSHIRTTOP': cname = 'T-shirt/top'
        tgt_name=cname.capitalize()
        text = f'{cname}\n(Target)'
    elif c==non_tgt:
        text = f"{cname.capitalize()}\n(Alternative)"
        fontweight="bold"
        fontsize=14
        non_tgt_name=cname.capitalize()
    else:
        text = cname.capitalize()
        fontweight="normal"
        fontsize=12
    ax.annotate(text, (x+0.5, y-0.1), 
                color='black', fontsize=fontsize, fontweight=fontweight,
                # ha="center", va="center",
                bbox = dict(facecolor='white', edgecolor='black', 
                            boxstyle='round,pad=0.3', alpha=0.6)
                )

ax.set_title(f'{tgt_name} vs. {non_tgt_name} - Accuracy: {accuracy:.2f}')
cbar = fig.colorbar(im, ax = ax, fraction=0.04, pad = 0.04)
cbar.set_label('Probability')
fig.savefig(os.path.join(str(Path(figpath).parent), f'pmap_{clean_discr_str}_75samples.{ext}'))

#%%
"""
PROBABILITY LINE PLOTS
"""
tgt, non_tgt = tgt_non_tgt
vae.target_size = 256
random_probability_dfs = {}
for metric in ['p', 'sigma']:
    metric_dfs = all_dfs[metric]
    for i, key in enumerate(trajectory_matrices.keys()):
        if not 'Random' in key: continue
        ALL_TRAJS = trajectory_matrices[key] # shape (501, 7781, 2)
        tnames = names_dfs[key]
        if ALL_TRAJS.shape[1]== 0: continue
    
        title = key + f' - {metric}(t)'
        clean_title = re.sub('[^a-zA-Z0-9]','', title)
        fname = os.path.join(figpath, f'heatmap_{clean_title}_NA')
        offset=2 if metric=='p' else 0
        if ('Random' in key) and (metric=='p'):
            random_probability_dfs[key] = get_probabilities(ALL_TRAJS, tgt, vae, discriminator, batch_size=1024)
            heatmap_mean_X_over_time(random_probability_dfs[key], ALL_TRAJS, tnames, latent_prototype, 
                                     title, save=True, fname=fname, offset=offset, ext=ext, N_A=10)
        else:
            heatmap_mean_X_over_time(metric_dfs[key], ALL_TRAJS, tnames, latent_prototype,
                                     title, save=True, fname=fname, offset=offset, ext=ext, N_A=10)

#%%
"""
TRAJECTORIES
"""
x_margin = 15
y_margin = 15
from mpl_toolkits.axes_grid1 import make_axes_locatable
for label, trajectory_matrix in tqdm(trajectory_matrices.items(),
                                     desc='plotting trajectories',
                                     total=len(list(trajectory_matrices.keys()))):
    urname, igdis = keys[label]
    # if 'memory' not in label: continue
    if label != key: continue
    n_time, n_samples, _ = trajectory_matrix.shape
    
    # Plot all trajectories in 2D space
    fig, ax = plt.subplots(figsize=(8, 8))
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    h, w = probability_map.shape
    for ax_ in [ax, ax2, ax3]:
        im = ax_.imshow(
            probability_map,
            extent=[xmin-x_margin, xmax+x_margin, ymin-y_margin, ymax+y_margin],
            origin="lower",
            cmap="viridis",
            vmin=0, vmax=1,
            # aspect=(w/h),
            alpha=1
        )
        
        
    data = np.transpose(np.array([pca_pipe_proto.transform(t) for t in
                     np.transpose(trajectory_matrix, axes = (1,0,2))]),
                        axes = (1,0,2))
    for i in range(n_samples):
        ax.plot(data[:, i, 0], data[:, i, 1], alpha=0.07, color='white')  # trajectory of sample i
        ax2.plot(data[:, i, 0], data[:, i, 1], alpha=0.05, color='white')  # trajectory of sample i

    # Overlay mean trajectories
    ALL_TRAJS = trajectory_matrices[label]
    TRAJS_groupbyR = {
        init: ALL_TRAJS[:, idx, :].mean(axis=1)   # shape (501, 2)
        for init, idx in names_dfs[label].groupby("init").groups.items()
        if idx.max() < ALL_TRAJS.shape[1]
    }
    
    # below: needed for colorbars
    colors = cm.inferno(np.linspace(0, 1, n_time))
    norm = mcolors.Normalize(vmin=0, vmax=n_time)
    sm = cm.ScalarMappable(cmap="inferno", norm=norm)
    sm.set_array([]) 
    
    for mean_traj_ in tqdm(TRAJS_groupbyR.values(),
                          total = len(list(TRAJS_groupbyR.values())),
                          desc = 'plotting mean trajs'):
        mean_traj = pca_pipe_proto.transform(mean_traj_)
        colors = cm.inferno(np.linspace(0, 1, n_time))
        for t in range(n_time - 1):
            ax.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2,  1], color=colors[t], linewidth=1.5) 
            ax2.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2,  1], color=colors[t], linewidth=1.5) 
            ax3.plot(mean_traj[t:t+2, 0], mean_traj[t:t+2,  1], color=colors[t], linewidth=1.5) 
    
    # Attach colorbars without shrinking
    for fig_,ax_ in zip([fig, fig2, fig3], [ax, ax2, ax3]):
        divider = make_axes_locatable(ax_)
        cax_right = divider.append_axes("right", size="5%", pad=0.1)
        cax_bottom = divider.append_axes("bottom", size="5%", pad=0.55)
        fig_.colorbar(sm, cax=cax_right, orientation="vertical", label='Time')
        fig_.colorbar(im, cax=cax_bottom, label=r"$p(y=y^\ast \mid x)$;  $x$: projection of $z$ to the native space of the data"+"\n"+r"(PCA visualization of $z$ coordinates)", location='bottom')
    for ax_ in [ax, ax2, ax3]:
        ax_.plot(pca_df_proto.iloc[tgt].PC1, pca_df_proto.iloc[tgt].PC2, markersize=20, marker = 'X', color='c', markeredgecolor='white', label=r"Target: $z : y=y^\ast$")
        ax_.plot(pca_df_proto.iloc[non_tgt].PC1, pca_df_proto.iloc[non_tgt].PC2, markersize=20, marker = 'o', color='c', markeredgecolor='white', label=r"Alternative: $z : y=y^{alt}$")
        ax_.set_xlabel(r'$PC_1$')
        ax_.set_ylabel(r'$PC_2$')
        ax_.legend(loc='upper right')
        

        ax_.set_xlim(xmin-x_margin, xmax+x_margin)
        ax_.set_ylim(ymin-y_margin, ymax+y_margin)
        ax_.set_aspect("equal")
    ax.grid(True)
    ax3.grid(True)   
    # plt.show()
    fig.savefig(os.path.join(figpath, f'trajs_{urname}_igdis{igdis}.{ext}'))
    fig2.savefig(os.path.join(figpath, f'trajs2_{urname}_igdis{igdis}.{ext}'))
    fig3.savefig(os.path.join(figpath, f'trajs3_{urname}_igdis{igdis}.{ext}'))
    
    plt.show()