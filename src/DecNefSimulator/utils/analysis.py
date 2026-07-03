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
from utils.utils import ReconstructionDataset, bidirectional_reduction

"""
PROBABILITY MAP OF THE GENERATOR LATENT SPACE USING THE CLASSIFIER
"""

@torch.no_grad()
def generator_probability_map(generator, z_dim, classifier, target_class_idx, n_samples,
                              init_points = None,
                              init_targets = None,
                              reduction='PCA',
                              space_radius = None
                              ):
    generator.eval()
    classifier.eval()
    if z_dim==2: return generator_probability_map_2d(generator, z_dim, classifier, target_class_idx, n_samples, space_radius)
    else: return generator_probability_map_(generator, z_dim, classifier, target_class_idx, n_samples, init_points, init_targets, reduction=reduction)

@torch.no_grad()
def generator_probability_map_(generator, z_dim, classifier, target_class_idx, n_samples,
                               init_points=None, init_targets = None, reduction='PCA'):
    # 1) PROJECT LATENT CLASS PROTOTYPES TO 2D USING PCA
    latent_prototypes = None
    prototypes = None
    labels = []
    generator.eval()
    device = classifier.device
    classifier.to(device)
    generator.to(device)    
    idx = 0 if target_class_idx==min(classifier.classes) else 1
    z0 = torch.tensor(init_points, device=device, dtype=torch.float)
    x0, *_ = generator.decoder(z0, generator.target_size)
    prototypes = x0.cpu() if prototypes is None else torch.cat((prototypes, x0.cpu()), dim=0)
    latent_prototypes = z0 if latent_prototypes is None else torch.cat((latent_prototypes, z0.cpu()), dim=0)
    labels = init_targets

    prototype_dataset = ReconstructionDataset(prototypes, latent_prototypes, labels)
    pca_pipe_proto, pca_df_proto = bidirectional_reduction(prototype_dataset, latent=True, dim=2, reduction='PCA')
    # 2) FIX BOUNDARIES FOR THE PLOT
    xmin, xmax = pca_df_proto.PC1.min(), pca_df_proto.PC1.max()
    ymin, ymax = pca_df_proto.PC2.min(), pca_df_proto.PC2.max()
    # 3) SAMPLE UNIFORMLY FROM THE GRID, compute probability map
    x_vals = np.linspace(xmin, xmax, n_samples)
    y_vals = np.linspace(ymin, ymax, n_samples)
    print(xmin, xmax, ymin, ymax)
    generated_samples = []
    coordinates = []
    probability_map = np.empty((n_samples, n_samples))
    for i, y in tqdm(enumerate(y_vals), total = len(y_vals)):
        for j, x in enumerate(x_vals):
            z = torch.Tensor(pca_pipe_proto.inverse_transform(np.array([x,y]).reshape(1, -1))).to(device)
            generated_sample = generator.decoder(z, target_size=generator.target_size)
            p = torch.nn.Softmax(dim=0)(
                classifier(
                    generated_sample
                    ).flatten()
                )[idx]
            generated_samples.append(generated_sample)
            probability_map[i, j] = p.cpu().numpy()
            coordinates.append([x, y])
    coordinates = np.array(coordinates)
    return probability_map, coordinates, generated_samples, pca_pipe_proto, pca_df_proto

@torch.no_grad() 
def generator_probability_map_2d(generator, z_dim, classifier, target_class_idx,
                                 n_samples, space_radius):
    xmin, xmax, ymin, ymax = space_radius
    x_vals = np.linspace(xmin, xmax, n_samples)
    y_vals = np.linspace(ymin, ymax, n_samples)
    generated_samples = []
    coordinates = []
    probability_map = np.empty((n_samples, n_samples))
    idx = 0 if target_class_idx==min(classifier.classes) else 1
    device = classifier.device
    classifier.to(device)
    generator.to(device)
    with torch.no_grad():
        # Iterate over the grid
        for i, y in tqdm(enumerate(y_vals),
                         total=len(y_vals)):
            for j, x in enumerate(x_vals):
                z = torch.Tensor((x,y)).to(device)
                # print(x, y, z.shape)
                try:
                    generated_sample = generator.decoder(z, generator.target_size)
                except:
                    generated_sample = generator.decoder_net(z.unsqueeze(0), generator.z_dim)
                if len(generated_sample.shape)<3:
                    generated_sample = generated_sample.unflatten(0, classifier.image_shape).unsqueeze(0)                   
                p = torch.nn.Softmax()(classifier(generated_sample).flatten())
                logits = classifier(generated_sample.to(device)).flatten()
                p_logits = torch.nn.functional.softmax(logits, dim=-1)
                p = p[idx]
                generated_samples.append(generated_sample)
                probability_map[i, j] = p.cpu().numpy()
                coordinates.append([x,y])
    coordinates = np.array(coordinates)
    pca_pipe_proto, pca_df_proto = None, None
    return probability_map, coordinates, generated_samples, pca_pipe_proto, pca_df_proto
@torch.no_grad()
def get_probabilities(z_np, target_class_idx, generator, classifier, batch_size=1024, device="cuda:0"):
    # Assume z_np is your latent array, shaped (Ntime, Ntraj, zdim)
    # generator.decoder: (N, zdim) -> image/voxels tensor
    # classifier: image/voxels tensor -> probability
    # Convert latent samples to torch
    z = torch.from_numpy(z_np.reshape(-1, generator.z_dim)).float().to(device)  # (Ntime*Ntraj, 2)
    idx = 0 if target_class_idx==min(classifier.classes) else 1
    probs = []
    generator.eval()
    classifier = classifier.to(device)
    generator =  generator.to(device)
    classifier.eval()
    
    with torch.no_grad():
        for i in trange(0, z.shape[0], batch_size):
            batch = z[i:i+batch_size]

            # Decode: latent -> image/fmri
            imgs = generator.decoder(batch, generator.target_size).to(device)
            # Classify: image/fmri -> probability
            logits = classifier(imgs)
            p =  torch.nn.Softmax(dim=1)(logits)[:,idx]
            probs.append(p.detach().cpu())
            
    # Concatenate all probabilities
    probs = torch.cat(probs).numpy()
    # Reshape back to (Ntime, Ntraj)
    return pd.DataFrame(probs.reshape(z_np.shape[:2]))

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def load_trajectory_npz(path):
    try:
        with np.load(path) as traj:
            return {
                'path': path,
                'probabilities': traj['probabilities'],
                'sigma': traj['sigma'],
                'trajectory': traj['trajectory'],
                # 'generated_images': traj['generated_images'],
            }
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
@torch.no_grad()
def process_metrics_for_traj(traj, prototype_tensor, latent_prototype, decnef_iters):
    try:
        probs = traj['probabilities']
        sigmas = traj['sigma']
        # np.squeeze was added because some new results are stored in shape (nTime, 1, z_dim)
        traj_array = np.squeeze(traj['trajectory'])
        z_dim = traj_array.shape[-1]

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

@torch.no_grad()
def trajectory_properties_as_df(trajectory_paths, decnef_iters, prototype, latent_prototype, verbose=False):
    n_trajs = len(trajectory_paths)
    z_dim = latent_prototype.shape[1]
    # Load all trajectories in parallel (I/O bound)
    with ThreadPoolExecutor(max_workers=16) as executor:
        if verbose:
            loaded_trajectories = list(tqdm(executor.map(load_trajectory_npz, trajectory_paths),
                                            total=n_trajs, desc="Loading trajectories"))

        else:
           loaded_trajectories = list(executor.map(load_trajectory_npz, trajectory_paths))

    # Track original indices
    valid_indices = [i for i, t in enumerate(loaded_trajectories) if t is not None]
    loaded_trajectories = [t for t in loaded_trajectories if t is not None]
    n_loaded = len(loaded_trajectories)

    # Prepare prototype tensor for GPU
    prototype_tensor = prototype.to(dtype=torch.float32, device=device)

    # Preallocate matrices
    probability_matrix = np.full((decnef_iters + 1, n_loaded), np.nan)
    sigma_matrix = np.full((decnef_iters + 1, n_loaded), np.nan)
    trajectory_matrix = np.full((decnef_iters + 1, n_loaded, z_dim), np.nan)

    # Process metrics sequentially
    for i, traj in enumerate(loaded_trajectories):
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

@torch.no_grad()
def latent_prototypes_to_fmri(all_class_prototypes, generator):
    fmri_prototypes = {}
    with torch.no_grad():
        for name, prototype_gaussian in all_class_prototypes.items():
            prototype = prototype_gaussian[0]
            prototype_fmri = generator.decoder(torch.Tensor(prototype).to(generator.device),
                                        generator.target_size).cpu().numpy()
            fmri_prototypes[name] = prototype_fmri
    return fmri_prototypes


