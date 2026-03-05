#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementing DecNef protocols used in 
DecNefSimulator: A Modular, Interpretable Framework for Decoded Neurofeedback Simulation Using Generative Models(Olza et al.)
https://arxiv.org/abs/2511.14555

Created on Wed Feb 19 11:39:45 2025

@author: alexolza
"""
import torch
import numpy as np
import random
from components.update_rules import powsig
 
def minimal_loop(train_loader, generator, discriminator,
                 target_class: int, lambda_, n_iter: int, device,  
                 update_rule_func, p_scale_func=powsig,#identity_f_p, 
                 z_current=None, 
                 ignore_discriminator:int = 0,
                 random_state:int=0, noise_sigma=torch.tensor(1.0),
                 warm_up:int = 2, stop_eps=1e-3, early_stopping:bool=False,
                 **update_rule_kwargs):
    """
    Parameters
    ----------
    train_loader : torch DataLoader
    generator : Latent Variable Generative model from components/generators.py
    discriminator : Feedback system from components/classifiers.py
    target_class : int. The target of DecNef training.
    lambda_ : TYPE
        DESCRIPTION.
    n_iter : int. Number of DecNef iterations
    device : Whether to work in CPU, GPU...
    update_rule_func : callable/function.
        DESCRIPTION: Encodes the artificial participant's decision-making
    p_scale_func : callable/function, optional (used if update_rule_func requires it)
    z_current : tensor, optional
        DESCRIPTION: Current cognitive state
    ignore_discriminator : bool/int, optional
        DESCRIPTION: Do DecNef training (0) or control experiment with sham feedback (1)
    random_state : int, optional. Reproducibility.
    noise_sigma : tensor, optional
        DESCRIPTION: Variance in the neural outcomes of regulation attempts.
    warm_up : int, optional: Stability parameter.
        DESCRIPTION: Window of observations from which the running average feedback is computed.
    stop_eps : float, optional. Convergence flag. Stop if variance goes below stop_eps
    early_stopping : bool, optional. Stopping flag.
    **update_rule_kwargs : Additional kwargs for the update rule

    Returns
    -------
    generated_images : List of np arrays. Observable trajectory in data space.
    trajectory: np array. Latent cognitive trajectory.
    probabilities: np.array. Feedback sequence
                  (computed in real time from the observable trajectory,
                   averaged in windows according to the value of warm_up)
    probabilities: np.array. Also includes probabilities from warm-up iterations.
    sigmas : np.array. Evolution of the variance in the neural outcomes of regulation attempts.

    """
    """
    Reproducibility and initialization
    """
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.cuda.manual_seed(random_state)
    generator.eval()
    z_dim = generator.z_dim
    # idx: The position of the item in the probability array
    # (i.e. [p, 1-p]: take p if target_class < non_target_class, else take 1-p)
    idx = 0 if target_class==min(discriminator.classes) else 1
    if z_current is None:
        z_current = torch.zeros(1,z_dim)
    # Ensure appropriate tensor shape
    if len(z_current.shape) == 1: z_current = z_current.unsqueeze(0)   
    X0 = generator.decoder(z_current.to(device), generator.target_size)    
            
    # FIRST ROUND WITH NEUTRAL FEEDBACK
    p = torch.Tensor([0.5])
    generated_images=[X0[0].cpu().detach().numpy()]
    probabilities = [p.item()]
    all_probabilities = [p.item()]
    trajectory = [z_current.numpy().flatten()] # This gathers moves that produce feedback
    all_trajectory = [z_current.numpy().flatten()] # This also considers warm-up moves
    sigmas = [noise_sigma]
    past_probabilities_mean = p.to(device)
    recent_probabilities_mean = p.to(device)
    patience0 = 25
    patience=patience0
    """
    Minimal DecNef loop
    """
    with torch.no_grad():
        for i in range(1, n_iter+1):
            if early_stopping:
                if i>=150: 
                    if (recent_probabilities_mean.item()>= 0.9) or sigmas[-1]<0.1:
                        patience-=1
                    else:
                        patience = patience0
                if patience==0: print(f'iter {i}, Early Stopping'); break
            
            warm_up_iters = 4*warm_up if i==1 else 2*warm_up
            
            # Warm-up iterations
            for j in range(2*warm_up_iters): 
                # There is no real-time feedback update during warm-up!
                # Note that recent_probabilities_mean and past_probabilities_mean are not being updated here
                z_new, _ = update_rule_func(all_trajectory, recent_probabilities_mean, past_probabilities_mean, lambda_, p_scale_func,
                                            noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                            warm_up=True,
                                            seed=j,
                                            **update_rule_kwargs)
                z_new = z_new.float()
                if len(z_new.shape) == 1: z_new = z_new.unsqueeze(0)
                x_decoded = generator.decoder(z_new.to(device), generator.target_size)
                all_trajectory.append(z_new.cpu().numpy().flatten())
                
                
                # THIS FEEDBACK IS NOT PROVIDED! It is only computed for analysis.
                if ignore_discriminator==0:
                    p =  torch.nn.Softmax()(discriminator(x_decoded).flatten())[idx]
                else:
                    # DecNef sham feedback is produced from a uniform distribution
                    # (for control experiments)
                    p = torch.rand(1)
                
                all_probabilities.append(p.item())
            # Compute average feedback after the warm-up iterations
            if i>1:
                recent_probabilities_mean = torch.tensor(np.mean(all_probabilities[-i*warm_up_iters:])).to(device)
                past_probabilities_mean = torch.tensor(np.mean(all_probabilities[-(i+1)*warm_up_iters: -i*warm_up_iters])).to(device)
            
            # Proceed to change cognitive state with the updated feedback
            z_new, noise_sigma = update_rule_func(trajectory, recent_probabilities_mean, past_probabilities_mean, lambda_, p_scale_func,
                                                    noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                                    warm_up=False,
                                                    seed=i,
                                                    **update_rule_kwargs)            
            sigmas.append(noise_sigma)
            z_new = z_new.float()
            if len(z_new.shape) == 1: z_new = z_new.unsqueeze(0)
            x_decoded = generator.decoder(z_new.to(device), generator.target_size)
            
            if ignore_discriminator==0:
                p =  torch.nn.Softmax()(discriminator(x_decoded).flatten())[idx]
            else:
                # DecNef sham feedback is produced from a uniform distribution
                # (for control experiments)
                p = torch.rand(1)
    
            all_probabilities.append(p.item())
            generated_images.append(x_decoded[0].cpu().detach().numpy())
            trajectory.append(z_new.cpu().numpy().flatten())
            all_trajectory.append(z_new.cpu().numpy().flatten())     
            probabilities.append(recent_probabilities_mean.item()) 
    sigmas = np.array(sigmas)
    return  generated_images, np.array(trajectory), np.array(probabilities), np.array(all_probabilities), sigmas
#%%
def compute_single_trajectory(vae, discriminator, trajectory_random_seed,
             train_loader, target_class, update_rule_func, p_scale_func, trajectory_name, 
                                     z_current = torch.Tensor([0,0]),
                                   n_iter = 15, lambda_ = 0.15, device='cuda', ignore_discriminator=0,
                                   **f_p_kwargs): 
    vae.eval()
    discriminator.eval()
    torch.manual_seed(trajectory_random_seed)
    random.seed(trajectory_random_seed)
    np.random.seed(trajectory_random_seed)
    torch.cuda.manual_seed(trajectory_random_seed)
    generated_images,\
    trajectory,\
    probabilities,\
    all_probabilities,\
    sigma  = minimal_loop(train_loader, vae, discriminator, 
                          target_class, lambda_, n_iter, device,
                          update_rule_func, p_scale_func,z_current,
                          ignore_discriminator=ignore_discriminator, 
                          random_state=trajectory_random_seed
                          )
    return generated_images, trajectory, probabilities, all_probabilities, sigma

 
