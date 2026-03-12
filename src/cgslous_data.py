import jax
import jax.numpy as jnp
from jax import lax
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpyro.infer import log_likelihood


def generate_synthetic_data(Nsub=10, K=2, D=10, steps=30):
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Define Total Observations
    N_total = Nsub * steps
    
    # Time and Covariates
    t_single = jnp.linspace(0, 5, steps)
    t = jnp.tile(t_single, Nsub)
    s_cov = jax.random.normal(k1, (Nsub,))
    
    start_idx = jnp.arange(0, N_total, steps)
    end_idx = start_idx + steps - 1
    
    # True Parameters (Mapping for comparison plots)
    theta_true = jnp.array([0.8, 1.5]) # Theta in image
    alpha_true = jnp.array([0.5, -0.5]) # alpha in image
    phi_true = jnp.array([1.2, -0.8])   # Phi in image
    gamma_true = jnp.array([0.3, 0.4]) # Gamma in image
    psi_true = jnp.full(D, 0.1)        # Psi in image
    
    # Identifiable Lambda (Lower Triangular Constraint)
    Lambda_true = jnp.zeros((D, K))
    Lambda_true = Lambda_true.at[0, 0].set(1.0)
    Lambda_true = Lambda_true.at[1, 1].set(1.2)
    Lambda_true = Lambda_true.at[2:, :].set(jax.random.normal(k2, (D-2, K)) * 0.5)
    
    # Generate Latent Paths f_i(t) 
    f_all = []
    for i in range(Nsub):
        fi = jnp.zeros((steps, K))
        # f_i(t) initial state
        mu_0 = alpha_true + phi_true * s_cov[i] * t_single[0] 
        fi = fi.at[0].set(mu_0 + jax.random.normal(k3, (K,)) * 0.1)
        
        for step in range(1, steps):
            dt = t_single[step] - t_single[step-1]
            mu_curr = alpha_true + phi_true * s_cov[i] * t_single[step]
            mu_prev = alpha_true + phi_true * s_cov[i] * t_single[step-1]
            
            # OU transition logic: df = -Theta(f - mu)dt + Gamma dW
            drift = jnp.exp(-theta_true * dt)
            mean = mu_curr + drift * (fi[step-1] - mu_prev)
            var = (gamma_true**2 / (2 * theta_true)) * (1 - jnp.exp(-2 * theta_true * dt))
            fi = fi.at[step].set(mean + jax.random.normal(k4, (K,)) * jnp.sqrt(var))
        f_all.append(fi)
    
    f_flat = jnp.concatenate(f_all, axis=0)
    # Observation model: x = Lambda * f + epsilon
    y = jnp.matmul(f_flat, Lambda_true.T) + jax.random.normal(k4, (N_total, D)) * psi_true
    
    return {
        "Nsub": Nsub, "N_total": N_total, "start_idx": start_idx,
        "end_idx": end_idx, "t": t, "y": y, "s_cov": s_cov, 
        "f_true": f_all,
        "true_params": {
            "Lambda": Lambda_true, 
            "theta_true": theta_true, 
            "alpha_true": alpha_true, 
            "phi_true": phi_true, 
            "gamma_true": gamma_true,
            "psi_true": psi_true
        }
    }


def generate_lou_only_data(Nsub=10, K=2, D=10, steps=30):
    key = jax.random.PRNGKey(77)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    N_total = Nsub * steps
    
    t_single = jnp.linspace(0, 5, steps)
    t = jnp.tile(t_single, Nsub)
    s_cov = jax.random.normal(k1, (Nsub,)) # Covariates exist but won't affect the mean
    
    start_idx = jnp.arange(0, N_total, steps)
    end_idx = start_idx + steps - 1
    
    # Parameters for a stationary process (Phi = 0)
    theta_true = jnp.array([1.2, 0.8])
    alpha_true = jnp.array([0.5, -0.5]) # Constant mean
    gamma_true = jnp.array([0.4, 0.3])
    psi_true = jnp.full(D, 0.1)
    
    # Non-sparse Lambda (Dense)
    Lambda_true = jax.random.normal(k2, (D, K)) * 0.5
    
    f_all = []
    for i in range(Nsub):
        fi = jnp.zeros((steps, K))
        # Process reverts to alpha_true regardless of time/covariates
        fi = fi.at[0].set(alpha_true + jax.random.normal(k3, (K,)) * 0.1)
        
        for step in range(1, steps):
            dt = t_single[step] - t_single[step-1]
            phi = jnp.exp(-theta_true * dt)
            mean = alpha_true + phi * (fi[step-1] - alpha_true)
            var = (gamma_true**2 / (2 * theta_true)) * (1 - jnp.exp(-2 * theta_true * dt))
            fi = fi.at[step].set(mean + jax.random.normal(k4, (K,)) * jnp.sqrt(var))
        f_all.append(fi)
    
    f_flat = jnp.concatenate(f_all, axis=0)
    y = jnp.matmul(f_flat, Lambda_true.T) + jax.random.normal(k4, (N_total, D)) * psi_true
    
    return {
        "Nsub": Nsub, "N_total": N_total, "start_idx": start_idx,
        "end_idx": end_idx, "t": t, "y": y, "s_cov": s_cov, 
        "f_true": f_all,
        "true_params": {
            "Lambda": Lambda_true, "theta_true": theta_true, 
            "alpha_true": alpha_true, "phi_true": jnp.zeros(K), # The trend is 0
            "gamma_true": gamma_true
        }
    }