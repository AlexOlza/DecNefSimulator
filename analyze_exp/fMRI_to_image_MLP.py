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
import re
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.plotting import plot_glass_brain
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os
import h5py
import random
import matplotlib.pyplot as plt
########################################
from nilearn.image import load_img, math_img
from analysis.utils import latent_prototypes_to_fmri
from components.generators import VAE
from utils import NPZDataset, ReconstructionDataset, bidirectional_reduction
from components.discriminators import ElasticNetLinearClassification

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

seed_everything(42)

subj=8
dataset = 'synth_fMRI_FASHION'
train_flag = True
repeat = 5
npz_file_path = f'../data/fMRIsynth/subj0{subj}/FASHION/full_dataset_repeat{repeat}_ae_plus_prior_pred_fmri_with_indices.npz'

transform = transforms.Compose([transforms.ToTensor()])
all_images = datasets.FashionMNIST('../data', download=True, train=train_flag, transform=transform)
npz_data = NPZDataset(npz_file_path, train=train_flag)
images = Subset(all_images, indices=npz_data.original_img_indices)
voxels = npz_data.data
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
EXP_NAME = 'synth_fMRI_FASHION_'
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
discriminator_type = 'CNN' if dataset=='FASHION' else 'ELASTICNET'
generator_name = 'VAE'
discriminator_epochs = 10
discriminator_batch_size = 16
tgt_non_tgt = [target_class_idx, non_target_class_idx]
tgt, non_tgt= tgt_non_tgt
tabular = True
n_features = npz_data.data.shape[-1]
modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/'

generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

vae = VAE(z_dim=z_dim,tabular=tabular, n_features=n_features, device= device).to(device)
vae.load(generator_fname+'.pt')
vae.eval()
vae.device= device
vae_history = vae.history_to_df()
print(f'Loaded {generator_fname}')


latent_prototype = vae.prototypes[target_class_idx][0] # [1] is the variance and [0] is the mu
alt_latent_prototype = vae.prototypes[non_target_class_idx][0]
prototype = vae.decoder(torch.Tensor(latent_prototype).to('cuda:1'),
                                            vae.target_size)
alt_prototype = vae.decoder(torch.Tensor(alt_latent_prototype).to('cuda:1'),
                                            vae.target_size)
#%%
all_class_prototypes = {classdict[k]: v for k,v in vae.prototypes.items()}
fmri_prototypes = latent_prototypes_to_fmri(all_class_prototypes, vae)

data_path = '../../external/MindSimulator/Codes/mindeye2_src'
func_data_path = f'../../external/MindEyeV2/data/subj0{subj}_func/'
betas_ = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
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


df = pd.DataFrame({k:v[0].ravel() for k,v in all_class_prototypes.items()})

scope = (df.min().min(), df.max().max())
samples = np.random.uniform(*scope, size=(10, 256))

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
subj = 8
img_size = 14386
class_name_dict_reverse = {v: k for k, v in class_name_dict.items()} 
combo_names = [class_name_dict_reverse[i] for i in tgt_non_tgt]
clean_discr_str = re.sub('[^a-zA-Z0-9]','', f'{combo_names[0]} vs {combo_names[1]}')
figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/{clean_discr_str}/nfb_eval/' if subj==8 else f'../EXPERIMENTS/{EXP_NAME}/{subj}/figures/{clean_discr_str}/nfb_eval/'

discriminator_name = f'{discriminator_type}_{clean_discr_str}__BS{discriminator_batch_size}_E{discriminator_epochs}'
discriminator_fname = os.path.join(modelpath, discriminator_name+'.pt')
discriminator = ElasticNetLinearClassification(img_size, tgt_non_tgt, device='cuda:1')

discriminator.load(discriminator_fname)
discriminator.eval()

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

# 1BIS) SANITY CHECKS
MSE = np.mean((pca_inv_proto - latent_prototypes.numpy())**2)
# sns.heatmap((pca_inv_proto - latent_prototypes.numpy())**2) # visual sanity check

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

#%%
# 4) PLOT MAP  of points (grid))
dx = xmax - xmin
dy = ymax - ymin

if dy < dx:
    mid = 0.5 * (ymin + ymax)
    ymin = mid - dx / 2
    ymax = mid + dx / 2

# scope = (df.min().min(), df.max().max())

# X_pca = pca_pipe_proto.transform(df.T)  # shape (n_samples, 256)
low_pca = [xmin, ymin]
high_pca = [xmax, ymax]
pca_points = np.random.uniform(low=low_pca,
                          high=high_pca,
                          size=(10, 2))


fig, ax = plt.subplots(1,1, figsize = (8, 8))

ax.set_xlabel(r'$PC_1$'+ '\n' + '(PCA visualization of $z$ coordinates)', labelpad=1)
ax.set_ylabel(r'$PC_2$', labelpad=0.1)
ax.grid(True)
ax.set_xlim(xmin-5, xmax+5)
ax.set_ylim(ymin-1, ymax+1)

for cname, c in class_name_dict.items():
    x, y = pca_df_proto.iloc[c][['PC1', 'PC2']]
    ax.plot(x, y, marker = 'X', color = 'red', markeredgecolor='white',  markersize=18, zorder=12)
    text = cname.capitalize()
    fontweight="bold"
    fontsize=14
    ax.annotate(text, (x+0.5, y-0.1), 
                color='black', fontsize=fontsize, fontweight=fontweight,
                # ha="center", va="center",
                bbox = dict(facecolor='white', edgecolor='black', 
                            boxstyle='round,pad=0.3', alpha=1)
                )


df = pd.DataFrame({k:v[0].ravel() for k,v in all_class_prototypes.items()})


for i, point_2d in enumerate(pca_points):
    ax.annotate(f'({i})', point_2d.ravel(), color="red",
    fontsize=10,
    # fontweight="bold",
    ha="center",
    va="center",
    bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.3", alpha=0.7)
    )
    
ax.set_title('Latent space of the VAE (internal cognitive states)')
fig.savefig('trial3_grid_points.pdf')
#%%
z_points = pca_pipe_proto.inverse_transform(pca_points)
img_values = []
for i,p in enumerate(z_points):
    fig_recon, axs_recon = plt.subplots(1, 1,figsize=(3,3))
    proto = vae.decoder(torch.Tensor(p).unsqueeze(0).to('cuda:1'),
                                                z_dim)
    
    # reconstruct one sample
    img_hat = reconstruct(model, proto, device)
    axs_recon.imshow(img_hat[0,0,:,:], cmap="gray")
    axs_recon.set_title('Semantic representation')
    axs_recon.axis("off")
    fig_recon.tight_layout()
    fig_recon.savefig(f'reconstruction_random_{i}.pdf')
    
img_values = np.concatenate(img_values)
img_values[img_values==0] = np.nan
global_max = np.nanpercentile(img_values, 99,)  # or 98 or 95
global_max = np.nanmax(img_values)
for i,p in enumerate(z_points):
    roi_img = nib.load(f'activations_{i}.nii.gz')
    fig_glass = plt.figure(figsize=(9, 3), facecolor='white')
    
    display = plot_glass_brain(roi_img,
                     # plot_abs = False, 
                     vmax = global_max,
                     annotate = False,
                     # axes = axs_glass,
                     figure = fig_glass,
                     # title = 'Observable space (fMRI pattern)',
                     # output_file = os.path.join(f'glass_{i}.pdf')
                     )
    fig_glass.suptitle('Observable space (fMRI pattern)')
    fig_glass.savefig(f'glass_{i}.pdf', bbox_inches='tight')
    plt.show()