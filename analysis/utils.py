#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models(Olza et al.)
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Fri Jul  4 12:06:42 2025

@author: alexolza
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import trange

"""
PROBABILITY MAP OF THE VAE LATENT SPACE USING THE TARGET DISCRIMINATOR
"""
def vae_probability_map(vae, discriminator, target_class_idx, n_samples=500, space_radius=2.5):
    x_vals = np.linspace(-space_radius, space_radius, n_samples)
    y_vals = np.linspace(-space_radius, space_radius, n_samples)
    generated_samples = []
    coordinates = []
    probability_map = np.empty((n_samples, n_samples))
    idx = 0 if target_class_idx==min(discriminator.classes) else 1

    with torch.no_grad():
        # Iterate over the grid
        for i, y in tqdm(enumerate(y_vals),
                         total=len(y_vals)):
            for j, x in enumerate(x_vals):
                z = torch.Tensor((x,y)).to(vae.device)
                try:
                    generated_sample = vae.decoder(z, vae.target_size)
                except ValueError:
                    generated_sample = vae.decoder(z.unsqueeze(0), vae.target_size)
                p = torch.nn.Softmax(dim=0)(discriminator(
                    generated_sample.to(discriminator.device)).flatten())
                logits = discriminator(generated_sample.to(discriminator.device)).flatten()
                p_logits = torch.nn.functional.softmax(logits, dim=0)
                if (j==0) and (i==0):
                    print(discriminator.classes)
                    print("idx:", idx)
                    print('discriminator(generated_sample).shape: ',discriminator(generated_sample).shape)
                    print('discriminator(generated_sample).flatten().shape: ',discriminator(generated_sample).flatten().shape)
                    print(f'p= {p}, p_logits = {p_logits}, logits = {logits}')
                p = p[idx]
                generated_samples.append(generated_sample)
                probability_map[i, j] = p.cpu().numpy()
                coordinates.append([x,y])
    return probability_map, coordinates, generated_samples

def get_probabilities(z_np, target_class_idx, vae, discriminator, batch_size=1024, device="cuda:0"):
    # Assume z_np is your latent array, shaped (Ntime, Ntraj, zdim)
    # vae.decoder: (N, zdim) -> image/voxels tensor
    # discriminator: image/voxels tensor -> probability
    # Convert latent samples to torch
    z = torch.from_numpy(z_np.reshape(-1, vae.target_size)).float().to(device)  # (Ntime*Ntraj, 2)
    idx = 0 if target_class_idx==min(discriminator.classes) else 1
    probs = []
    vae.eval()
    discriminator = discriminator.to(device)
    vae = vae.to(device)
    discriminator.eval()
    
    with torch.no_grad():
        for i in trange(0, z.shape[0], batch_size):
            batch = z[i:i+batch_size]

            # Decode: latent -> image/fmri
            imgs = vae.decoder(batch, vae.target_size).to(device)
            # Classify: image/fmri -> probability
            logits = discriminator(imgs)
            p =  torch.nn.Softmax(dim=1)(logits)[:,idx]
            probs.append(p.detach().cpu())
            
    # Concatenate all probabilities
    probs = torch.cat(probs).numpy()
    # print(probs.shape)
    # Reshape back to (Ntime, Ntraj)
    return pd.DataFrame(probs.reshape(z_np.shape[:2]))

def get_multiclass_probabilities(z_np, classes, vae, discriminator, batch_size=1024, device="cuda:0"):
    # Assume z_np is your latent array, shaped (Ntime, Ntraj, zdim)
    # vae.decoder: (N, zdim) -> image/voxels tensor
    # discriminator: image/voxels tensor -> probabilities (Nclass)
    n_time, n_trans, z_dim = z_np.shape
    # Convert latent samples to torch
    z = torch.from_numpy(z_np.reshape(-1, z_dim)).float().to(device)  # (Ntime*Ntraj, zdim)
    probs = []
    vae.eval()
    discriminator = discriminator.to(device)
    vae = vae.to(device)
    discriminator.eval()
    
    with torch.no_grad():
        for i in trange(0, n_time, batch_size):
            batch = z[i:i+batch_size]

            # Decode: latent -> image
            imgs = vae.decoder(batch, z_dim).to(device)
            # Classify: image -> probabilities
            logits = discriminator(imgs)
            p =  torch.nn.Softmax(dim=1)(logits)
            probs.append(p.detach().cpu())
            # break
    # Concatenate all probabilities
    probs = torch.cat(probs).numpy()
    return pd.DataFrame(probs.reshape(z_np.shape[:2]))

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trajectory_npz(path):
    try:
        with np.load(path) as traj:
            return {
                'path': path,
                'probabilities': traj['probabilities'],
                'sigma': traj['sigma'],
                'trajectory': traj['trajectory'],
                'generated_images': traj['generated_images'],
            }
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def process_metrics_for_traj(traj, prototype_tensor, latent_prototype, decnef_iters):
    try:
        probs = traj['probabilities']
        sigmas = traj['sigma']
        traj_array = traj['trajectory']
        z_dim = traj_array.shape[1]

        L = min(decnef_iters + 1, len(probs))

        result = {
            'prob': np.full(decnef_iters + 1, np.nan),
            'sigma': np.full(decnef_iters + 1, np.nan),
            'traj': np.full((decnef_iters + 1, z_dim), np.nan),
        }

        result['prob'][:L] = probs[:L]
        result['sigma'][:L] = sigmas[:L]
        result['traj'][:L] = traj_array[:L]

        return result

    except Exception as e:
        print(f"Metric processing error in {traj['path']}: {e}")
        return None


def trajectory_properties_as_df(trajectory_paths, decnef_iters, prototype, latent_prototype):
    n_trajs = len(trajectory_paths)
    z_dim = latent_prototype.shape[1]
    # Load all trajectories in parallel (I/O bound)
    with ThreadPoolExecutor(max_workers=16) as executor:
        loaded_trajectories = list(tqdm(executor.map(load_trajectory_npz, trajectory_paths),
                                        total=n_trajs, desc="Loading trajectories"))

    # Track original indices
    valid_indices = [i for i, t in enumerate(loaded_trajectories) if t is not None]
    loaded_trajectories = [t for t in loaded_trajectories if t is not None]
    n_loaded = len(loaded_trajectories)

    # Prepare prototype tensor for GPU
    prototype_tensor = torch.tensor(prototype, dtype=torch.float32, device=device)

    # Preallocate matrices
    probability_matrix = np.full((decnef_iters + 1, n_loaded), np.nan)
    sigma_matrix = np.full((decnef_iters + 1, n_loaded), np.nan)
    trajectory_matrix = np.full((decnef_iters + 1, n_loaded, z_dim), np.nan)

    # Process metrics sequentially
    for i, traj in enumerate(tqdm(loaded_trajectories, desc="Computing metrics on GPU")):
        result = process_metrics_for_traj(traj, prototype_tensor, latent_prototype, decnef_iters)
        if result:
            probability_matrix[:, i] = result['prob']
            sigma_matrix[:, i] = result['sigma']
            trajectory_matrix[:, i] = result['traj']

    # Create DataFrames
    probability_df = pd.DataFrame(probability_matrix)
    sigma_df = pd.DataFrame(sigma_matrix)
    trajectory_names = [trajectory_paths[i].split('/')[-1] for i in valid_indices]
    names_df = pd.DataFrame({'traj_name': trajectory_names})

    # Assign correct column names to probability_df and others
    probability_df.columns = names_df['traj_name']
    sigma_df.columns = names_df['traj_name']
    return probability_df, sigma_df, trajectory_matrix, names_df


def latent_prototypes_to_fmri(all_class_prototypes, vae):
    fmri_prototypes = {}
    with torch.no_grad():
        for name, prototype_gaussian in all_class_prototypes.items():
            prototype = prototype_gaussian[0]
            prototype_fmri = vae.decoder(torch.Tensor(prototype).to(vae.device),
                                        vae.target_size).cpu().numpy()
            fmri_prototypes[name] = prototype_fmri
    return fmri_prototypes

