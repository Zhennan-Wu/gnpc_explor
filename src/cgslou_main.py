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
import argparse
from contextlib import redirect_stdout

from cgslou_models import LOU, CGSLOU
from cgslous_data import generate_synthetic_data, generate_lou_only_data
from cgslou_analysis import (plot_parameter_comparison, plot_lou_parameter_comparison, plot_all_traces, plot_lou_traces, plot_latent_recovery, plot_lou_latent_recovery, calculate_recovery_metrics, calculate_lou_recovery_metrics, compare_models, plot_sparsity_histograms, plot_residual_comparison)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def run_comparison_pipeline(K=2, D=10, Nsub=12, steps=30, num_warmup=500, num_samples=1000, save_dir=None):
    # --- 1. Generate Data ---
    data = generate_synthetic_data(Nsub=Nsub, K=K, D=D, steps=steps)
    input_data = {k: v for k, v in data.items() if k not in ['true_params', 'f_true']}
    y_obs = data["y"]
    
    rng_key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(rng_key)

    # --- 2. Fit Models ---
    print("Fitting Original LOU...")
    model_lou = LOU(K=K, D=D)
    samples_lou = model_lou.fit(k1, num_warmup=num_warmup, num_samples=num_samples, **input_data)

    print("Fitting New CGSLOU...")
    model_cgslou = CGSLOU(K=K, D=D)
    samples_cgslou = model_cgslou.fit(k2, num_warmup=num_warmup, num_samples=num_samples, **input_data)

    # --- 3. Manual ELPD Calculation (Robust to scan errors) ---
    def calculate_manual_elpd(samples, y_true):
        # Reconstruct predictions: y_hat = x_trace @ Lambda^T
        # samples['x_trace'] is (num_samples, N_total, K)
        # samples['Lambda'] is (num_samples, D, K)
        
        # We take the mean over samples to get expected log likelihood
        x_mean = jnp.mean(samples['x_trace'], axis=0)
        L_mean = jnp.mean(samples['Lambda'], axis=0)
        
        # Use inferred observation noise
        # LOU uses sigma_obs, CGSLOU uses psi_diag
        obs_sd = jnp.mean(samples.get('psi_diag', samples.get('sigma_obs')), axis=0)
        
        mu_pred = jnp.matmul(x_mean, L_mean.T)
        
        # Log-probability of a Normal distribution
        log_probs = dist.Normal(mu_pred, obs_sd).log_prob(y_true)
        return jnp.sum(log_probs)

    print("\n" + "="*50)
    print("              FINAL COMPARISON METRICS")
    print("="*50)

    # MSE Accuracy
    f_true_flat = jnp.concatenate(data["f_true"], axis=0)
    mse_lou = jnp.mean(jnp.square(f_true_flat - jnp.mean(samples_lou['x_trace'], axis=0)))
    mse_cgs = jnp.mean(jnp.square(f_true_flat - jnp.mean(samples_cgslou['x_trace'], axis=0)))

    # ELPD Scores
    elpd_lou = calculate_manual_elpd(samples_lou, y_obs)
    elpd_cgs = calculate_manual_elpd(samples_cgslou, y_obs)

    print(f"{'Metric':<25} | {'Original LOU':<15} | {'New CGSLOU':<15}")
    print("-" * 60)
    print(f"{'Latent State MSE':<25} | {mse_lou:<15.6f} | {mse_cgs:<15.6f}")
    print(f"{'Total Log-Likelihood':<25} | {elpd_lou:<15.2f} | {elpd_cgs:<15.2f}")
    
    return samples_lou, samples_cgslou, data, model_lou, model_cgslou


def run_stationary_comparison(Nsub=10, K=2, D=10, steps=30, num_warmup=500, num_samples=1000, save_dir=None):
    # Generate data with NO trend
    print("Generating stationary data (no trend)...")
    data = generate_lou_only_data(Nsub=Nsub, K=K, D=D, steps=steps)
    input_data = {k: v for k, v in data.items() if k not in ['true_params', 'f_true']}
    
    rng_key = jax.random.PRNGKey(88)
    k1, k2 = jax.random.split(rng_key)

    print("Fitting Original LOU...")
    model_lou = LOU(K=2, D=10)
    samples_lou = model_lou.fit(k1, num_warmup=num_warmup, num_samples=num_samples, **input_data)

    print("Fitting CGSLOU (Should identify zero trend)...")
    model_cgs = CGSLOU(K=2, D=10)
    samples_cgs = model_cgs.fit(k2, num_warmup=num_warmup, num_samples=num_samples, **input_data)

    print("\n" + "="*50)
    print("              FINAL COMPARISON METRICS")
    print("="*50)

    # Recovery Check
    est_phi = jnp.mean(samples_cgs['phi_latent'], axis=0)
    print("\n" + "-"*40)
    print(f"CGSLOU Estimated Trend (Phi): {est_phi}")
    print("True Trend (Phi):             [0.0, 0.0]")
    print("-"*40)

    # Metric Comparison
    f_true = jnp.concatenate(data["f_true"], axis=0)
    mse_lou = jnp.mean(jnp.square(f_true - jnp.mean(samples_lou['x_trace'], axis=0)))
    mse_cgs = jnp.mean(jnp.square(f_true - jnp.mean(samples_cgs['x_trace'], axis=0)))

    print(f"LOU Latent MSE:    {mse_lou:.6f}")
    print(f"CGSLOU Latent MSE: {mse_cgs:.6f}")
    
    return samples_lou, samples_cgs, data, model_lou, model_cgs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo script")

    parser.add_argument(
        "--data",
        type=str,
        default="shifted",
        help="Data type (default: shifted)"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10000,
        help="Number of warmup steps (default: 5000)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of MCMC samples (default: 10000)"
    )

    args = parser.parse_args()

    print("data generation type", args.data)
    print("warmup:", args.warmup)
    print("samples:", args.samples)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    if args.data == "shifted":
        print("Running comparison on shifted data...")
        save_dir = "results/shifted_comparison/warmup_{}_samples_{}".format(args.warmup, args.samples)
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "comparison_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write(f"Data Type: Shifted\nWarmup Steps: {args.warmup}\nMCMC Samples: {args.samples}\n\n")
            log_file.flush()

            sys.stdout = Tee(sys.stdout, log_file)
            sys.stderr = sys.stdout  # optional
            samples_lou, samples_cgs, data, model_lou, model_cgs = run_comparison_pipeline(num_warmup=args.warmup, num_samples=args.samples, save_dir=save_dir)
            print(f"\n {"="*50} \n")
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr        
        # 3. Visual Diagnostics
        plot_parameter_comparison(samples_cgs, data, save_dir=save_dir) # Mirrored Heatmaps
        # Run the LOU comparison
        plot_lou_parameter_comparison(samples_lou, data, save_dir=save_dir)

        plot_all_traces(samples_cgs, save_dir=save_dir)                 # MCMC Convergence
        plot_lou_traces(samples_lou, save_dir=save_dir)

        plot_latent_recovery(samples_cgs, data, 0, save_dir=save_dir)   # Path Recovery
        plot_lou_latent_recovery(samples_lou, data, subject_idx=0, save_dir=save_dir)
        with open(log_path, "a") as log_file:
            log_file.flush()

            sys.stdout = Tee(sys.stdout, log_file)
            sys.stderr = sys.stdout  # optional
            calculate_recovery_metrics(samples_cgs, data, save_dir=save_dir)
            print(f"\n {"="*50} \n")
            calculate_lou_recovery_metrics(samples_lou, data, save_dir=save_dir)
            print(f"\n {"="*50} \n")
            compare_models(samples_cgs, samples_lou, data, save_dir=save_dir)          # Direct Comparison
            print(f"\n {"="*50} \n")
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        plot_sparsity_histograms(samples_cgs, samples_lou, save_dir=save_dir)
        plot_residual_comparison(samples_cgs, samples_lou, data, save_dir=save_dir)
    elif args.data == "stationary":
        print("Running stationary comparison pipeline...")
        save_dir = "results/stationary_comparison/warmup_{}_samples_{}".format(args.warmup, args.samples)
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "comparison_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write(f"Data Type: Stationary\nWarmup Steps: {args.warmup}\nMCMC Samples: {args.samples}\n\n")
            log_file.flush()

            sys.stdout = Tee(sys.stdout, log_file)
            sys.stderr = sys.stdout  # optional
            samples_lou, samples_cgs, data, model_lou, model_cgs = run_stationary_comparison(num_warmup=args.warmup, num_samples=args.samples, save_dir=save_dir)
            print(f"\n {"="*50} \n")
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        # 3. Visual Diagnostics
        plot_parameter_comparison(samples_cgs, data, save_dir=save_dir) # Mirrored Heatmaps
        # Run the LOU comparison
        plot_lou_parameter_comparison(samples_lou, data, save_dir=save_dir)

        plot_all_traces(samples_cgs, save_dir=save_dir)                 # MCMC Convergence
        plot_lou_traces(samples_lou, save_dir=save_dir)

        plot_latent_recovery(samples_cgs, data, 0, save_dir=save_dir)   # Path Recovery
        plot_lou_latent_recovery(samples_lou, data, subject_idx=0, save_dir=save_dir)
        with open(log_path, "a") as log_file:
            log_file.flush()

            sys.stdout = Tee(sys.stdout, log_file)
            sys.stderr = sys.stdout  # optional
            calculate_recovery_metrics(samples_cgs, data, save_dir=save_dir)
            print(f"\n {"="*50} \n")
            calculate_lou_recovery_metrics(samples_lou, data, save_dir=save_dir)
            print(f"\n {"="*50} \n")
            compare_models(samples_cgs, samples_lou, data, save_dir=save_dir)          # Direct Comparison
            print(f"\n {"="*50} \n")
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        plot_sparsity_histograms(samples_cgs, samples_lou, save_dir=save_dir)
        plot_residual_comparison(samples_cgs, samples_lou, data, save_dir=save_dir)

    else:
        print("Invalid data type. Use 'shifted' or 'stationary'.")
