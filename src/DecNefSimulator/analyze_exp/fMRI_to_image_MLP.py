#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure for
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Thu Mar  5 14:52:41 2026

@author: alexolza
"""
import sys
sys.path.append('..')
from utils.configuration import seed_everything
seed_everything(42)
import re
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.plotting import plot_glass_brain
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
########################################
from utils.utils import make_init_z_lattice
from nilearn.image import load_img, math_img, new_img_like
from utils.analysis import latent_prototypes_to_fmri, generator_probability_map
from components.generators import VAE
from utils.utils import NPZDataset
from components.classifiers import ElasticNetLinearClassification
from visualization.plotting import plot_probability_map_grid
from visualization.utils import VoxelImageDataset, VoxelToImage, train, reconstruct
#%%

subj=8
dataset = 'synth_fMRI_FASHION'
train_flag = True
repeat = 5
npz_file_path = f'../../../../data/fMRIsynth/subj0{subj}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri_with_indices.npz'

transform = transforms.Compose([transforms.ToTensor()])
all_images = datasets.FashionMNIST('../data', download=True, train=train_flag, transform=transform)
npz_data = NPZDataset(npz_file_path, train=train_flag)
images = Subset(all_images, indices=npz_data.original_img_indices)
voxels = npz_data.data
#%%
img_per_class = []

for i in range(10):
    idx = np.argmax(all_images.targets==i).item()
    img_per_class.append((i, idx, all_images.data[idx]))
    plt.imshow(all_images.data[idx]); plt.axis('off')
    plt.tight_layout(); plt.savefig(f'class{i}.png', dpi = 600, transparent = True)
    
#%%
classdict = {v: k for k, v in npz_data.class_to_idx.items()}
ncol = 8 #repeat - 1
nrow = 8
fig, axs = plt.subplots(nrow, ncol, figsize=(6,6))


for i in range(nrow * ncol):
    img, imglabel = images[i]
    idx = npz_data.original_img_indices[i]
    y = npz_data.targets[i]
    # yold = old_data.targets[i]
    classname = classdict[imglabel]
    ax = axs.ravel()[i]
    title = f'{idx}: {imglabel} - {y}'
    ax.imshow(img[0])
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()

#%%


images = torch.stack([t[0] for t in images])
dataset = VoxelImageDataset(images, voxels)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = VoxelToImage(n_voxels=voxels.shape[-1]).to(device)

train(model, loader, device, epochs=30)

# reconstruct one sample
img_hat = reconstruct(model, voxels[0], device)

plt.figure(figsize=(4,2))

plt.subplot(1,2,1)
plt.title("GT")
plt.imshow(images[0,0], cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Reconstruction")
plt.imshow(img_hat[0,0], cmap="gray")
plt.axis("off")

plt.show()

#%%
EXP_NAME = 'FASHION_fmri4'
target_class_idx = 0
non_target_class_idx = 1
decnef_iters =  500
n_trajs = 100
linv = 5
z_dim = 256
lambda_ = 1/linv 
generator_epochs = 25
space_radius=2.5 if dataset=='FASHION' else 3.5
n_samples=75
device='cuda:1'
generator_batch_size=64
classifier_type = 'CNN' if dataset=='FASHION' else 'ELASTICNET'
generator_name = 'VAE'
classifier_epochs = 10
classifier_batch_size = 16
tgt_non_tgt = [target_class_idx, non_target_class_idx]
tgt, non_tgt= tgt_non_tgt
tabular = True
n_features = npz_data.data.shape[-1]
modelpath = f'../EXPERIMENTS/{EXP_NAME}/subj8/weights/'

generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

generator = VAE(z_dim=z_dim,tabular=tabular, n_features=n_features, device= device).to(device)
generator.load(generator_fname+'.pt')
generator.eval()
generator.device= device
vae_history = generator.history_to_df()
print(f'Loaded {generator_fname}')


latent_prototype = generator.prototypes[target_class_idx][0] # [1] is the variance and [0] is the mu
alt_latent_prototype = generator.prototypes[non_target_class_idx][0]
prototype = generator.decoder(torch.Tensor(latent_prototype).to('cuda:1'),
                                            generator.target_size)
alt_prototype = generator.decoder(torch.Tensor(alt_latent_prototype).to('cuda:1'),
                                            generator.target_size)
#%%
all_class_prototypes = {classdict[k]: v for k,v in generator.prototypes.items()}
fmri_prototypes = latent_prototypes_to_fmri(all_class_prototypes, generator)

data_path = '../../../../external/MindSimulator/Codes/mindeye2_src'
func_data_path = f'../../../../external/MindEyeV2/data/subj0{subj}_func/'
beta_fname = '../../../../external/MindSimulator/Codes/mindeye2_src/betas_all_subj08_fp32_renorm.hdf5'
betas_ = h5py.File(beta_fname, 'r')
betas = betas_['betas'][:]
betas = torch.Tensor(betas).to("cpu").float()
# num_voxels = fmri_data.shape[-1]
mask = load_img(func_data_path + 'brainmask1pt8.nii.gz') # shape (80, 103, 78)


L_mask = load_img(func_data_path+'lh.nsdgeneral.nii.gz')
R_mask = load_img(func_data_path+'rh.nsdgeneral.nii.gz')
mask = math_img('img1 + img2', img1=R_mask, img2=L_mask)


mask_data = mask.get_fdata()
affine = mask.affine

# Find ROI voxels (== 0)
roi_mask = mask_data == 0
# Prepare empty full-volume
shape_3d = mask_data.shape
if betas.ndim == 1:
    full_brain = np.zeros(shape_3d)
    full_brain[roi_mask] = betas
    img = nib.Nifti1Image(full_brain, affine)
else:
    # For multiple beta maps; shape = (X, Y, Z, n_betas)
    n_betas = betas.shape[0]
    full_brain = np.zeros(shape_3d)#np.zeros(shape_3d + (n_betas,))
    for i in range(n_betas):
        # full_brain[..., i][roi_mask] = betas[i]
        full_brain[roi_mask] = betas[i]
        break
    img = nib.Nifti1Image(full_brain, affine)
   

df = pd.DataFrame({k:v[0].ravel() for k,v in all_class_prototypes.items()})

scope = (df.min().min(), df.max().max())
samples = np.random.uniform(*scope, size=(10, 256))

class_name_dict = {'T-shirt/Top': 0,
 'Trouser': 1,
 'Pullover': 2,
 'Dress': 3,
 'Coat': 4,
 'Sandal': 5,
 'Shirt': 6,
 'Sneaker': 7,
 'Bag': 8,
 'Ankleboot': 9} # Override it because of capitalization

subj = 8
img_size = 14386
class_name_dict_reverse = {v: k for k, v in class_name_dict.items()} 
combo_names = [class_name_dict_reverse[i] for i in tgt_non_tgt]
clean_discr_str = re.sub('[^a-zA-Z0-9]','', f'{combo_names[0].upper()} vs {combo_names[1].upper()}')
figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/{clean_discr_str}/nfb_eval/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/figures/{clean_discr_str}/nfb_eval/'

classifier_name = f'{classifier_type}_{clean_discr_str}__BS{classifier_batch_size}_E{classifier_epochs}'
classifier_fname = os.path.join(modelpath, classifier_name+'.pt')
classifier = ElasticNetLinearClassification(img_size, tgt_non_tgt, device='cuda:1')

classifier.load(classifier_fname)
classifier.eval()

"""
PROBABILITY MAP EQUIVALENT
"""
pca_fname = os.path.join(modelpath, "pca_latent_visualization.pkl")
pmap_fname = os.path.join(modelpath, f"pmap_{clean_discr_str}.npz")

n_samples=75
label='With memory (ELASTICNET)'
generator = generator.to(generator.device)
all_class_prototypes = np.vstack([prot[0].ravel() for idx, prot in generator.prototypes.items()
                                   ])
all_class_prototypes_sigma = np.vstack([prot[1].ravel() for idx, prot in generator.prototypes.items()
                                   ])

#%%
points = make_init_z_lattice(100000, z_dim, all_class_prototypes, all_class_prototypes_sigma, tgt_non_tgt, 
                            lattice_fname='aux', z_grid_init_fname='aux.npy')
with torch.no_grad():
        probability_map, coordinates, generated_samples, pca_pipe_proto, pca_df_proto = generator_probability_map(generator, z_dim, 
                                                                                                              classifier, target_class_idx,
                                                                                                              n_samples, space_radius = space_radius, 
                                                                                                              
                                                                                                              init_targets = points[:,-1],
                                                                                                              
                                                                                                              init_points = points[:,:z_dim],
                                                                                                              )

    
prots = pca_pipe_proto.transform(np.array(list(generator.prototypes.values()))[:,0,:,:].squeeze(1))
space_radius= (pca_df_proto.PC1.min(),pca_df_proto.PC1.max(), pca_df_proto.PC2.min(), pca_df_proto.PC1.max())
ext='png'
seed=42

plot_probability_map_grid(probability_map, coordinates, pca_pipe_proto,
                                     generator, z_dim, classifier,
                                     target_class_idx, non_target_class_idx, class_name_dict, clean_discr_str,
                                     n_samples,
                                     space_radius= (pca_df_proto.PC1.min(),pca_df_proto.PC1.max(), pca_df_proto.PC2.min(), pca_df_proto.PC1.max()),
                                     ext = ext,
                                     seed=seed,
                                     maxmin=True,
                                     accuracy= f'{classifier.history_to_df()["val_acc"].values[-1]:.2f}'
                                       )
    
space_radius= (pca_df_proto.PC1.min(),pca_df_proto.PC1.max(), pca_df_proto.PC2.min(), pca_df_proto.PC1.max())
cols = df.columns
plt.show()


pca_points = pca_df_proto.groupby('Class').sample(1, random_state=42)[['PC1', 'PC2']]
z_points = pca_pipe_proto.inverse_transform(pca_points)

fig, ax = plt.subplots(1,1)
ax.imshow(
    probability_map,
    extent=(pca_df_proto.PC1.min(),pca_df_proto.PC1.max(), pca_df_proto.PC2.min(), pca_df_proto.PC1.max()), #(-0.2,0.3, -0.2, 0.3),#(-0.2,0.3, -0.2, 0.3),
    origin="lower",
    cmap="viridis",
    vmin=0, vmax=1,
    alpha=0
)
for i in range(len(prots)):
    ax.scatter(prots[i,0], prots[i, 1], marker = 'X') 
    ax.annotate(f'{class_name_dict_reverse[i]}', (prots[i,0], prots[i, 1]), weight='bold')
for i, point in enumerate(pca_points.values):
    ax.annotate(f'({i})', (point[0], point[1]), bbox = dict(facecolor='white', edgecolor='black', 
                boxstyle='round,pad=0.4', alpha=0.9))
ax.set_title('Latent space of the VAE (internal cognitive states)')
ax.grid(True)
fig.savefig('vae_grid_points.pdf')


#%%
img_values = []
for i,p in enumerate(z_points.values):
    if i==0: print(p.shape)
    fig_recon, axs_recon = plt.subplots(1, 1,figsize=(3,3))
    voxels_i = generator.decoder(torch.Tensor(p).unsqueeze(0).to('cuda:1'),
                                                z_dim)
    
    # reconstruct one sample
    img_hat = reconstruct(model, voxels_i, device)
    axs_recon.imshow(img_hat[0,0,:,:], cmap="gray")
    axs_recon.set_title('Semantic representation')
    axs_recon.axis("off")
    fig_recon.tight_layout()
    fig_recon.savefig(f'reconstruction_random_{i}.pdf')
#%%    
with torch.no_grad():
    func = load_img(func_data_path + 'func1pt8-to-MNI.nii.gz') 
    for i,point in enumerate(z_points.values):
        # roi_img = nib.load(f'activations_{i}.nii.gz')
        voxels_i = generator.decoder(torch.Tensor(point).reshape(1,z_dim).to('cuda:1'),
                                                    generator.target_size)
        img_hat = reconstruct(model, voxels_i, device)
        voxels_i = voxels_i.cpu().numpy().reshape(14386)
        full_brain = np.zeros(shape_3d)
        full_brain[roi_mask] = voxels_i#betas
        img = nib.Nifti1Image(full_brain, affine)
        fig_glass = plt.figure(figsize=(9, 3), facecolor='white')        
        display = plot_glass_brain(img,
                         display_mode = 'z',
                         annotate = False,
                         colorbar=False,
                         figure = fig_glass,
                         )
        fig_glass.savefig(f'GLASS__{i}.png', bbox_inches='tight', dpi = 600)
        plt.show()
        

        ns = np.ceil(np.sqrt(voxels_i.size)).astype(int)
        s = np.zeros(ns**2)
        s[:voxels_i.size] = voxels_i*10
        s = s.reshape(ns,ns)
        smoothed = s.reshape(
            30, 4,
            30, 4
        ).mean(axis=(1,3))
        plt.imshow(smoothed, cmap='gist_rainbow')#'seismic'
        plt.axis('off')
        plt.savefig(f'HEATMAP2__{i}.png', bbox_inches='tight', dpi = 600, transparent = True)
        
        fig_recon, axs_recon = plt.subplots(1, 1,figsize=(3,3))
        axs_recon.imshow(img_hat[0,0,:,:], cmap="viridis")
        axs_recon.axis("off")
        fig_recon.tight_layout()
        fig_recon.savefig(f'REC__{i}.png', bbox_inches='tight', dpi = 600, transparent = True)
