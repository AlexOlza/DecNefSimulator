#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementing update rules (a.k.a. learning strategies) used in 
DecNefLab: A Modular and Interpretable Simulation Framework for Decoded Neurofeedback
(Olza et al.)
https://arxiv.org/abs/2511.14555

Refer to the paper above for detailed explanations.

Created on Tue Jun 10 15:58:15 2025

@author: alexolza
"""
import torch
from torch.distributions import MultivariateNormal
@torch.no_grad()
def powsig(p, p0,k=2, eps=1e-3, sigma0=1): 
    eps = torch.max(p0, p)
    scale = ((1-p)**2)*((p0+eps)/(p+eps))**k
    return scale
@torch.no_grad()
def update_z_moving_normal_drift_adaptive_variance(trajectory, p, p0,
                                                   lambda_val, f_p,
                                                   device,
                                                   warm_up=False,  
                                                   max_sigma=1, noise_sigma_0 =1,
                                                   seed=0, **f_p_kwargs):
    """
    LEARNING STRATEGY. ASSUMPTIONS:
        1) Variability in the neural outcomes of regulation attempts.
        2) Reward increase fosters exploitation. Reward decrease fosters exploration.
  
    Parameters
    ----------
    trajectory : Sequence of tensors. Cognitive trajectory.
    p : float. Latest feedback value.
    p0 : float. Preceeding feedback value.
    lambda_val : float. Trust-in-feedback parameter.
    f_p : function/callable. Adapts exploration/exploitation scale.
    delta : float between 0 and 1. Feedback worsening rejection threshold.
    warm_up : bool. Whether it's a DecNef warmup iteration.
            If True, new feedback won't be provided
    device : str, optional. CPU, GPU...
    max_sigma : float, optional. Max variance in neural outcomes of regulation attempts.
    noise_sigma_0 : Current variance in neural outcomes of regulation attempts.
    seed : int, optional. Reproducibility.
    verbose : bool, optional.
    **f_p_kwargs : Additional kwargs for learning rule.

    Returns
    -------
    z_new : New cognitive state.
    noise_sigma : New variance in neural outcomes of regulation attempts.

    """
    noise_sigma = ((1-lambda_val) *noise_sigma_0 + lambda_val * f_p(p, p0, **f_p_kwargs))
    noise_sigma = torch.minimum(
                    torch.as_tensor(noise_sigma, device=device),
                    torch.as_tensor(max_sigma, device=device),                    
                ).to(device)
    noise_cov = noise_sigma*torch.eye(trajectory[-1].shape[-1], device = device)
    noise_dist = MultivariateNormal(trajectory[-1], covariance_matrix=noise_cov)
    normal_update = noise_dist.sample().to(device)
    z_new =  (1-lambda_val) * trajectory[-1].to(device) + lambda_val * normal_update
    return z_new, noise_sigma

@torch.no_grad()
def update_z_moving_normal_drift_adaptive_variance_memory(trajectory, p, p0,
                                                          lambda_val, 
                                                          f_p,
                                                          device,
                                                          delta = 0.75,
                                                          warm_up = False,
                                                          max_sigma=1, noise_sigma_0 =1, seed=0, verbose=False, **f_p_kwargs):
    """
    LEARNING STRATEGY. ASSUMPTIONS:
        1) Variability in the neural outcomes of regulation attempts.
        2) Reward increase fosters exploitation. Reward decrease fosters exploration.
        3) Short term memory: Significant drop in feedback causes reversal to previous state.
        
    Parameters
    ----------
    trajectory : Sequence of tensors. Cognitive trajectory.
        DESCRIPTION.
    p : float. Latest feedback value.
    p0 : float. Preceeding feedback value.
    lambda_val : float. Trust-in-feedback parameter.
    f_p : function/callable. Adapts exploration/exploitation scale.
    delta : float between 0 and 1. Feedback worsening rejection threshold.
    warm_up : bool. Whether it's a DecNef warmup iteration.
            If True, new feedback won't be provided
    device : str, optional. CPU, GPU...
    max_sigma : float, optional. Max variance in neural outcomes of regulation attempts.
    noise_sigma_0 : Current variance in neural outcomes of regulation attempts.
    seed : int, optional. Reproducibility.
    verbose : bool, optional.
    **f_p_kwargs : Additional kwargs for learning rule.

    Returns
    -------
    z_new : New cognitive state.
    noise_sigma : New variance in neural outcomes of regulation attempts.
    """
    reverse = True if ((p<delta*p0).item() and (not warm_up)) else False
    if reverse: # This will be true if z_{i+1} is significantly worse than z_i
        if verbose: print(f'Reversal; p/p0={p/p0}')
        # Since z_{i+1} was bad, we return to z_i and we adopt a more exploratory strategy
        noise_sigma = torch.minimum(
                        torch.as_tensor(noise_sigma_0, device=device),
                        torch.as_tensor(max_sigma, device=device)
                    )
        noise_cov = noise_sigma*torch.eye(trajectory[-1].shape[-1], device = device)
    
        with torch.random.fork_rng():
            torch.manual_seed(seed)  # Local seed
            noise_dist = MultivariateNormal(trajectory[-2], covariance_matrix=noise_cov)
            normal_update = noise_dist.sample()#.to(device)
        z_new =  (1-lambda_val) * trajectory[-2].to(device) + lambda_val * normal_update
    else:
        z_new, noise_sigma = update_z_moving_normal_drift_adaptive_variance(trajectory, p, p0, lambda_val, f_p, device, warm_up, max_sigma, noise_sigma_0, **f_p_kwargs)
    return z_new, noise_sigma
