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
@torch.no_grad()
def minimal_loop(train_loader, generator, classifier,
                 target_class: int, lambda_, n_iter: int, device:str, 
                 update_rule_func, p_scale_func, #identity_f_p, 
                 z_current=None, 
                 ignore_classifier:int = 0,
                 random_state:int=0, noise_sigma=1.0,
                 warm_up:int = 2,
                 **update_rule_kwargs):
    """
    Parameters
    ----------
    train_loader : torch DataLoader
    generator : Latent Variable Generative model from components/generators.py
    classifier : Feedback system from components/classifiers.py
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
    ignore_classifier : bool/int, optional
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
    z_current = z_current.float().to(device)
    # idx: The position of the item in the probability array
    # (i.e. [p, 1-p]: take p if target_class < non_target_class, else take 1-p)
    idx = 0 if target_class==min(classifier.classes) else 1
    if z_current is None:
        z_current = torch.zeros(1,z_dim).float()
    # Ensure appropriate tensor shape
    if len(z_current.shape) == 1: z_current = z_current.unsqueeze(0)   
    X0 = generator.decoder(z_current, generator.target_size)    
            
    # FIRST ROUND WITH NEUTRAL FEEDBACK
    p = torch.tensor(0.5, device = device)
    generated_images=[X0[0].detach()]
    probabilities = [p.detach().squeeze()]
    all_probabilities = [p.detach().squeeze()]
    trajectory = [z_current] # This gathers moves that produce feedback
    all_trajectory = [z_current] # This also considers warm-up moves
    sigmas = [torch.tensor(noise_sigma, device=device).detach().squeeze()]
    past_probabilities_mean = p.to(device)
    recent_probabilities_mean = p.to(device)
    """
    Minimal DecNef loop
    """
    with torch.no_grad():
        for i in range(1, n_iter+1):  
            warm_up_iters = 4*warm_up if i==1 else 2*warm_up           
            # Warm-up iterations
            for j in range(2*warm_up_iters): 
                # There is no real-time feedback update during warm-up!
                # Note that recent_probabilities_mean and past_probabilities_mean are not being updated here

                z_new, _ = update_rule_func(all_trajectory, recent_probabilities_mean, past_probabilities_mean, 
                                            lambda_,
                                            p_scale_func, device, 
                                            noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                            warm_up=True,
                                            seed=j,
                                            **update_rule_kwargs)
                z_new = z_new.float()
                if len(z_new.shape) == 1: z_new = z_new.unsqueeze(0)
                x_decoded = generator.decoder(z_new.to(device), generator.target_size)
                all_trajectory.append(z_new)
                
                
                # THIS FEEDBACK IS NOT PROVIDED! It is only computed for analysis.
                if ignore_classifier==0:
                    try: 
                        p =  torch.nn.Softmax(dim=0)(classifier(x_decoded).flatten())[idx]
                    except RuntimeError: 
                        x_decoded = x_decoded.unflatten(1, classifier.image_shape)
                        p =  torch.nn.Softmax(dim=0)(classifier(x_decoded.to(device)).flatten())[idx]

                else:
                    # DecNef sham feedback is produced from a uniform distribution
                    # (for control experiments)
                    p = torch.rand(1, device=device)[0]
                
                all_probabilities.append(p.detach())
            # Compute average feedback after the warm-up iterations
            if i>1:
                recent_probabilities_mean = torch.stack( all_probabilities[-i*warm_up_iters:]).mean()
                past_probabilities_mean = torch.stack(all_probabilities[-(i+1)*warm_up_iters: -i*warm_up_iters]).mean()
            # Proceed to change cognitive state with the updated feedback
            z_new, noise_sigma = update_rule_func(trajectory, recent_probabilities_mean, past_probabilities_mean, lambda_, p_scale_func,
                                                    device=device, noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                                    warm_up=False,
                                                    seed=i,
                                                    **update_rule_kwargs)            
            sigmas.append(noise_sigma.detach().squeeze())
            z_new = z_new.float()
            if len(z_new.shape) == 1: z_new = z_new.unsqueeze(0)
            x_decoded = generator.decoder(z_new.to(device), generator.target_size)
            
            if ignore_classifier==0:
                try: 
                    p =  torch.nn.Softmax(dim=0)(classifier(x_decoded).flatten())[idx]
                except RuntimeError: 
                    x_decoded = x_decoded.unflatten(1, classifier.image_shape)
                    p =  torch.nn.Softmax(dim=0)(classifier(x_decoded.to(device)).flatten())[idx]
            else:
                # DecNef sham feedback is produced from a uniform distribution
                # (for control experiments)
                p = torch.rand(1, device=device)[0]
                
            generated_images.append(x_decoded[0].detach())
            trajectory.append(z_new.detach())
            all_trajectory.append(z_new.detach())     
            probabilities.append(recent_probabilities_mean.detach().squeeze()) 
            all_probabilities.append(p.detach().squeeze())
            
    sigmas = torch.stack(sigmas).cpu().numpy()
    generated_images = torch.stack(generated_images).cpu().numpy()
    trajectory = torch.stack(trajectory).cpu().numpy()
    probabilities = torch.stack(probabilities).cpu().numpy()
    all_probabilities = torch.stack(all_probabilities).cpu().numpy()
    return  generated_images, trajectory, probabilities, all_probabilities, sigmas
#%%
@torch.no_grad()
def compute_single_trajectory(vae, classifier, trajectory_random_seed,
                              train_loader, target_class, device,
                              update_rule_func, p_scale_func,
                              trajectory_name, z_current,
                              n_iter, lambda_, ignore_classifier,
                              **f_p_kwargs): 
    vae.eval()
    classifier.eval()
    torch.manual_seed(trajectory_random_seed)
    random.seed(trajectory_random_seed)
    np.random.seed(trajectory_random_seed)
    torch.cuda.manual_seed(trajectory_random_seed)
    z_current = torch.tensor(z_current, device=device)
    
    generated_images,\
    trajectory,\
    probabilities,\
    all_probabilities,\
    sigma  = minimal_loop(train_loader, vae, classifier, 
                          target_class, lambda_, n_iter, device,
                          update_rule_func, p_scale_func,z_current,
                          ignore_classifier=ignore_classifier, 
                          random_state=trajectory_random_seed
                          )
    return generated_images, trajectory, probabilities, all_probabilities, sigma

 
