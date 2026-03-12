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
import sys
import os


def plot_parameter_comparison(samples, data, save_dir=None):
    """
    Compares estimated parameters against ground truth via mirrored heatmaps.
    Updated to match the simulation keys: theta_true, alpha_true, etc.
    """
    true_p = data.get("true_params", {})
    if not true_p:
        print("No ground truth parameters found.")
        return

    # 1. Extract Posterior Means from MCMC samples
    est_lambda = jnp.mean(samples['Lambda'], axis=0)
    est_theta  = jnp.mean(samples['theta_diag'], axis=0)
    est_alpha  = jnp.mean(samples['alpha'], axis=0)
    est_phi    = jnp.mean(samples['phi_latent'], axis=0)
    est_gamma  = jnp.mean(samples['gamma'], axis=0)

    # 2. Map Ground Truth keys from your simulation function
    # In your simulation: theta_true, alpha_true, phi_true, gamma_true
    try:
        true_theta = true_p.get('theta_true', true_p.get('theta'))
        true_alpha = true_p.get('alpha_true', true_p.get('alpha'))
        true_phi   = true_p.get('phi_true', true_p.get('phi'))
        true_gamma = true_p.get('gamma_true', true_p.get('gamma'))
        
        # Verify no None values before stacking
        for name, val in zip(["theta", "alpha", "phi", "gamma"], [true_theta, true_alpha, true_phi, true_gamma]):
            if val is None:
                raise ValueError(f"Ground truth parameter '{name}' not found in data['true_params']")

        true_trans = jnp.stack([true_theta, true_alpha, true_phi, true_gamma])
        est_trans  = jnp.stack([est_theta, est_alpha, est_phi, est_gamma])
    except Exception as e:
        print(f"Error: {e}")
        return

    label_y = [r"$\Theta$ (Reversion)", r"$\alpha$ (Intercept)", r"$\Phi$ (Trend)", r"$\Gamma$ (Vol)"]
    label_x = [f"Factor {i}" for i in range(est_lambda.shape[1])]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loadings (Lambda)
    sns.heatmap(true_p['Lambda'], annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0, 0])
    axes[0, 0].set_title(r"True $\Lambda$ (Ground Truth)")
    sns.heatmap(est_lambda, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0, 1])
    axes[0, 1].set_title(r"Estimated $\Lambda$ (Posterior Mean)")

    # Transition (OU)
    sns.heatmap(true_trans, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=label_x, yticklabels=label_y, ax=axes[1, 0])
    axes[1, 0].set_title(r"True OU Parameters")
    sns.heatmap(est_trans, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=label_x, yticklabels=label_y, ax=axes[1, 1])
    axes[1, 1].set_title(r"Estimated OU Parameters")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "parameter_comparison.png"))
    plt.show()
    plt.close()


def plot_all_traces(samples, save_dir=None):
    """
    Plots MCMC traces for all main model parameters to diagnose convergence.
    """
    # Filter to main scalar/vector parameters
    params_to_plot = ['theta_diag', 'alpha', 'phi_latent', 'gamma', 'psi_diag']
    
    n = len(params_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    
    for i, name in enumerate(params_to_plot):
        if name in samples:
            vals = samples[name]
            if vals.ndim > 1:
                # Plot each component (e.g., K dimensions)
                for k in range(vals.shape[1]):
                    axes[i].plot(vals[:, k], alpha=0.6, label=f"Dim {k}")
            else:
                axes[i].plot(vals, alpha=0.8)
                
            axes[i].set_title(f"MCMC Trace: {name}")
            axes[i].legend(loc='upper right', fontsize='x-small')
            axes[i].set_ylabel("Value")

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "cgslou_all_traces.png"))
    plt.show()
    plt.close()


def plot_latent_recovery(samples, data, subject_idx=0, save_dir=None):
    """
    Plots the posterior mean of the latent states f_i(t) against the true paths.
    """
    true_f = data.get("f_true") # Requires f_true to be saved in generate_synthetic_data
    if true_f is None:
        print("True latent paths 'f_true' not found in data.")
        return

    # Extract trace (num_samples, N_total, K)
    x_trace = samples.get("x_trace")
    if x_trace is None:
        print("x_trace not found in samples. Ensure numpyro.deterministic('x_trace', ...) is in the model.")
        return

    est_f = jnp.mean(x_trace, axis=0)
    
    # Subject indices
    s, e = data["start_idx"][subject_idx], data["end_idx"][subject_idx]
    sub_t = data["t"][s:e+1]
    
    plt.figure(figsize=(10, 5))
    for k in range(est_f.shape[1]):
        plt.plot(sub_t, true_f[subject_idx][:, k], 'k--', alpha=0.5, label=f"True Factor {k}" if k==0 else "")
        plt.plot(sub_t, est_f[s:e+1, k], label=f"Est Factor {k}")
        
    plt.title(f"Latent Path Recovery (Subject {subject_idx})")
    plt.xlabel("Time (t)")
    plt.ylabel("Latent State $f(t)$")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"cgslou_latent_recovery_subject_{subject_idx}.png"))
    plt.show()
    plt.close()


def calculate_recovery_metrics(samples, data, save_dir=None):
    """
    Calculates MSE for latent paths and key model parameters.
    """
    true_p = data.get("true_params", {})
    f_true = jnp.concatenate(data.get("f_true"), axis=0) # Flattened ground truth
    
    # 1. Latent Path MSE
    # x_trace is (num_samples, N_total, K)
    est_f = jnp.mean(samples['x_trace'], axis=0)
    mse_latent = jnp.mean(jnp.square(f_true - est_f))
    
    # 2. Parameter MSEs
    est_lambda = jnp.mean(samples['Lambda'], axis=0)
    mse_lambda = jnp.mean(jnp.square(true_p['Lambda'] - est_lambda))
    
    est_phi = jnp.mean(samples['phi_latent'], axis=0)
    mse_phi = jnp.mean(jnp.square(true_p['phi_true'] - est_phi))
    
    est_theta = jnp.mean(samples['theta_diag'], axis=0)
    mse_theta = jnp.mean(jnp.square(true_p['theta_true'] - est_theta))

    print("-" * 30)
    print("RECOVERY METRICS (MSE)")
    print("-" * 30)
    print(f"Latent States (f_i(t)): {mse_latent:.6f}")
    print(f"Factor Loadings (Lambda): {mse_lambda:.6f}")
    print(f"Trend Coeffs (Phi):      {mse_phi:.6f}")
    print(f"Reversion (Theta):      {mse_theta:.6f}")
    print("-" * 30)


def plot_sparsity_histograms(samples_cgs, samples_lou, save_dir=None):
    plt.figure(figsize=(10, 5))
    
    # Flatten all entries in the loading matrices
    l_cgs = jnp.mean(samples_cgs['Lambda'], axis=0).flatten()
    l_lou = jnp.mean(samples_lou['Lambda'], axis=0).flatten()
    
    plt.hist(l_lou, bins=50, alpha=0.5, label='Original LOU (Gaussian Prior)', color='gray')
    plt.hist(l_cgs, bins=50, alpha=0.5, label='CGSLOU (Global-Local Prior)', color='blue')
    
    plt.axvline(0, color='black', linestyle='--')
    plt.title(r"Distribution of Factor Loadings ($\Lambda$ entries)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "sparsity_histograms.png"))
    plt.show()
    plt.close()


def compare_models(samples_cgslou, samples_lou, data, save_dir=None):
    """
    Compares the New (CGSLOU) vs Original (LOU) on ground truth recovery.
    """
    f_true = jnp.concatenate(data["f_true"], axis=0)
    Lambda_true = data["true_params"]["Lambda"]

    # Posterior Means
    f_cgs = jnp.mean(samples_cgslou['x_trace'], axis=0)
    f_lou = jnp.mean(samples_lou['x_trace'], axis=0)
    
    L_cgs = jnp.mean(samples_cgslou['Lambda'], axis=0)
    # Assuming LOU also produces a Lambda or W matrix
    L_lou = jnp.mean(samples_lou['Lambda'], axis=0) 

    # Calculate MSE
    mse_f_cgs = jnp.mean(jnp.square(f_true - f_cgs))
    mse_f_lou = jnp.mean(jnp.square(f_true - f_lou))
    
    mse_L_cgs = jnp.mean(jnp.square(Lambda_true - L_cgs))
    mse_L_lou = jnp.mean(jnp.square(Lambda_true - L_lou))

    print(f"{'Metric':<20} | {'Original LOU':<15} | {'New CGSLOU':<15}")
    print("-" * 55)
    print(f"{'Latent State MSE':<20} | {mse_f_lou:<15.6f} | {mse_f_cgs:<15.6f}")
    print(f"{'Loading Matrix MSE':<20} | {mse_L_lou:<15.6f} | {mse_L_cgs:<15.6f}")


def plot_residual_comparison(samples_cgslou, samples_lou, data, save_dir=None):
    y = data["y"]
    # Reconstruct predictions
    pred_cgs = jnp.matmul(jnp.mean(samples_cgslou['x_trace'], axis=0), jnp.mean(samples_cgslou['Lambda'], axis=0).T)
    pred_lou = jnp.matmul(jnp.mean(samples_lou['x_trace'], axis=0), jnp.mean(samples_lou['Lambda'], axis=0).T)
    
    res_cgs = y - pred_cgs
    res_lou = y - pred_lou
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data["t"], jnp.mean(res_lou, axis=1), alpha=0.3, label="LOU Residuals")
    plt.title("Original LOU Residuals (Bias Visible)")
    plt.subplot(1, 2, 2)
    plt.scatter(data["t"], jnp.mean(res_cgs, axis=1), alpha=0.3, color='green', label="CGSLOU Residuals")
    plt.title("CGSLOU Residuals (Clean)")
    plt.xlabel("Time (t)")
    plt.ylabel("Residual")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "residual_comparison.png"))
    plt.show()
    plt.close()


def plot_lou_parameter_comparison(samples_lou, data, save_dir=None):
    """
    Compares the original LOU estimated parameters against ground truth.
    Highlights the lack of sparsity and the stationary mean assumption.
    """
    true_p = data.get("true_params", {})
    if not true_p:
        print("No ground truth parameters found.")
        return

    # 1. Extract Posterior Means from LOU samples
    est_lambda = jnp.mean(samples_lou['Lambda'], axis=0)
    est_theta  = jnp.mean(samples_lou['theta'], axis=0)
    est_mu0    = jnp.mean(samples_lou['mu0'], axis=0)
    # LOU has no alpha, phi, or gamma_vol as defined in the new model logic
    
    # 2. Extract Ground Truth (mapped to LOU's simpler structure)
    true_theta = true_p.get('theta_true')
    true_mu0   = true_p.get('alpha_true') # In simulation, alpha is the intercept/baseline
    
    # We create a 2-row comparison for the transition parameters
    true_trans = jnp.stack([true_theta, true_mu0])
    est_trans  = jnp.stack([est_theta, est_mu0])

    label_y = [r"$\Theta$ (Reversion)", r"$\mu_0$ (Constant Target)"]
    label_x = [f"Factor {i}" for i in range(est_lambda.shape[1])]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Lambda (Loadings) Comparison ---
    # Note: LOU will likely show non-zero values where truth is zero
    sns.heatmap(true_p['Lambda'], annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0, 0])
    axes[0, 0].set_title(r"True $\Lambda$ (Ground Truth Sparse)")
    
    sns.heatmap(est_lambda, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0, 1])
    axes[0, 1].set_title(r"LOU Estimated $\Lambda$ (Dense/Noisy)")

    # --- Transition (Stationary) Comparison ---
    sns.heatmap(true_trans, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=label_x, yticklabels=label_y, ax=axes[1, 0])
    axes[1, 0].set_title("True Reversion & Baseline")

    sns.heatmap(est_trans, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=label_x, yticklabels=label_y, ax=axes[1, 1])
    axes[1, 1].set_title("LOU Estimated Reversion & Baseline")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "lou_parameter_comparison.png"))
    plt.show()
    plt.close()


def calculate_lou_recovery_metrics(samples_lou, data, save_dir=None):
    """
    Calculates MSE for latent paths and available parameters in the LOU model.
    """
    true_p = data.get("true_params", {})
    f_true_flat = jnp.concatenate(data.get("f_true"), axis=0)
    
    # 1. Latent Path MSE
    est_f_lou = jnp.mean(samples_lou['x_trace'], axis=0)
    mse_latent = jnp.mean(jnp.square(f_true_flat - est_f_lou))
    
    # 2. Loading Matrix MSE
    est_lambda_lou = jnp.mean(samples_lou['Lambda'], axis=0)
    mse_lambda = jnp.mean(jnp.square(true_p['Lambda'] - est_lambda_lou))
    
    # 3. Transition MSE (Comparing mu0 to the true alpha/intercept)
    est_theta = jnp.mean(samples_lou['theta'], axis=0)
    mse_theta = jnp.mean(jnp.square(true_p['theta_true'] - est_theta))

    print("-" * 35)
    print("ORIGINAL LOU RECOVERY METRICS")
    print("-" * 35)
    print(f"Latent States (f_i(t)): {mse_latent:.6f}")
    print(f"Factor Loadings (Lambda): {mse_lambda:.6f}")
    print(f"Reversion (Theta):       {mse_theta:.6f}")
    print("-" * 35)


def plot_lou_traces(samples_lou, save_dir=None):
    """
    Plots MCMC traces for the original LOU parameters.
    """
    param_names = ['theta', 'mu0', 'sigma0', 'sigma_obs']
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 2 * n_params), sharex=True)
    
    for i, name in enumerate(param_names):
        data_plot = samples_lou[name]
        if data_plot.ndim > 1:
            for k in range(data_plot.shape[1]):
                axes[i].plot(data_plot[:, k], alpha=0.7, label=f"Factor {k}")
        else:
            axes[i].plot(data_plot, alpha=0.7)
            
        axes[i].set_title(f"LOU Trace: {name}")
        axes[i].set_ylabel("Value")
        if i == 0: axes[i].legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "lou_traces.png"))
    plt.show()
    plt.close()


def plot_lou_latent_recovery(samples_lou, data, subject_idx=0, save_dir=None):
    """
    Plots LOU's estimated latent states against the true paths.
    """
    true_f = data.get("f_true")
    est_f_lou = jnp.mean(samples_lou['x_trace'], axis=0)
    
    s, e = data["start_idx"][subject_idx], data["end_idx"][subject_idx]
    sub_t = data["t"][s:e+1]
    
    plt.figure(figsize=(10, 5))
    for k in range(est_f_lou.shape[1]):
        plt.plot(sub_t, true_f[subject_idx][:, k], 'k--', alpha=0.4, label=f"True Factor {k}" if k==0 else "")
        plt.plot(sub_t, est_f_lou[s:e+1, k], label=f"LOU Est Factor {k}")
        
    plt.title(f"LOU Latent Path Recovery (Subject {subject_idx})")
    plt.xlabel("Time (t)")
    plt.ylabel("Latent State $f(t)$")
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"lou_latent_recovery_subject_{subject_idx}.png"))
    plt.show()
    plt.close()
