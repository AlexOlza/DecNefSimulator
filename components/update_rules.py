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

def powsig(p, p0,k=2, eps=1e-3, sigma0=1, reverse = False): 
    eps = torch.max(p0.cpu(),p.cpu())
    scale = ((1-p.cpu())**2)*((p0.cpu()+eps)/(p.cpu()+eps))**k
    return scale.item()

def update_z_moving_normal_drift(z, p, target_dist, lambda_val, f_p, device='cuda', noise_sigma =torch.tensor([[1.0,1.0]]), **f_p_kwargs):
    """Another LEARNING STRATEGY. Not used in the paper."""
    assert noise_sigma.shape==z.shape, print('shape mismatch: ', noise_sigma.shape, z.shape)
    noise_cov = noise_sigma*torch.eye(z.shape[-1]).cpu()
    noise_dist = MultivariateNormal(z.cpu(), covariance_matrix=noise_cov)
    t = target_dist.sample().to(device)
    det_upd_normal =  f_p(p, reverse=False, **f_p_kwargs).to(device) * t
    normal_update = f_p(p, reverse=True, **f_p_kwargs).to(device)* noise_dist.sample().to(device)
    z_new = (1-lambda_val) *  (normal_update +z.to(device)) + lambda_val * det_upd_normal
    return z_new

def update_z_moving_normal_drift_adaptive_variance(trajectory, p, p0,
                                                   lambda_val, f_p,
                                                   warm_up=False, 
                                                   device='cuda', 
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
    z  = torch.tensor(trajectory[-1])
    noise_sigma = ((1-lambda_val) *noise_sigma_0 + lambda_val * f_p(p.cpu(), p0.cpu(), **f_p_kwargs)).cpu()
    noise_sigma = torch.tensor(min(noise_sigma.item(), max_sigma)).cpu()
    noise_cov = noise_sigma*torch.eye(z.shape[-1]).cpu()
    noise_dist = MultivariateNormal(z.cpu(), covariance_matrix=noise_cov)
    normal_update = noise_dist.sample().to(device)
    z_new =  (1-lambda_val) * z.to(device) + lambda_val * normal_update
    return z_new, noise_sigma

def update_z_moving_normal_drift_adaptive_variance_memory(trajectory, p, p0,
                                                          lambda_val, 
                                                          f_p,
                                                          delta = 0.75,
                                                          warm_up = False,
                                                          device='cuda', max_sigma=1, noise_sigma_0 =1, seed=0, verbose=False, **f_p_kwargs):
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
    z  = torch.tensor(trajectory[-1])
    reverse = True if ((p<delta*p0) and (not warm_up)) else False
    if reverse: # This will be true if z_{i+1} is significantly worse than z_i
        if verbose: print(f'Reversal; p/p0={p/p0}')
        z_previous = torch.tensor(trajectory[-2])    
        # Since z_{i+1} was bad, we return to z_i and we adopt a more exploratory strategy
        noise_sigma = torch.tensor(min(noise_sigma_0.item(),
                                       max_sigma)).cpu()
        noise_cov = noise_sigma*torch.eye(z.shape[-1]).cpu()
        with torch.random.fork_rng():
            torch.manual_seed(seed)  # Local seed
            noise_dist = MultivariateNormal(z_previous.cpu(), covariance_matrix=noise_cov)
            normal_update = noise_dist.sample().to(device)
        z_new =  (1-lambda_val) * z_previous.to(device) + lambda_val * normal_update
    else:
        z_new, noise_sigma = update_z_moving_normal_drift_adaptive_variance(trajectory, p, p0, lambda_val, f_p, warm_up, device, max_sigma, noise_sigma_0, **f_p_kwargs)
    return z_new, noise_sigma