import os
import json
import glob
import time
import numpy as np
import pandas as pd
import cmdstanpy
from scipy.linalg import expm
from scipy.special import expit
import argparse
import concurrent.futures
import multiprocessing
from functools import reduce
from scipy.linalg import solve_continuous_lyapunov
import arviz as az
import re


# ==========================================
# 1. Data Generation (Updated to K=12, q=4)
# ==========================================
def generate_lou_simulation_data(N=600, scenario='L2G3C4', random_state=42):
    """
    Generates data for the Latent Ornstein-Uhlenbeck (LOU) simulation study.
    """
    np.random.seed(random_state)

    # Measurement Model (IRT) Parameters
    # 12 items (1-5 binary, 6-12 ordinal), 4 latent dimensions
    # Structured to load primarily onto 1 specific factor for identification
    if 'L1' in scenario:
        Lambda = np.array([
            [0.8, 0.0, 0.0, 0.0], [1.2, 0.0, 0.0, 0.0], [1.1, 0.0, 0.0, 0.0], # Latent 1
            [0.0, 0.9, 0.0, 0.0], [0.0, 1.4, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], # Latent 2
            [0.0, 0.0, 0.7, 0.0], [0.0, 0.0, 1.1, 0.0], [0.0, 0.0, 0.9, 0.0], # Latent 3
            [0.0, 0.0, 0.0, 1.2], [0.0, 0.0, 0.0, 0.8], [0.0, 0.0, 0.0, 1.0]  # Latent 4
        ])
    elif 'L2' in scenario:
        Lambda = np.array([
            [1.2, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [4.1, 0.0, 0.0, 0.0], # Latent 1
            [0.0, 3.1, 0.0, 0.0], [0.0, 5.2, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], # Latent 2
            [0.0, 0.0, 1.7, 0.0], [0.0, 0.0, 2.4, 0.0], [0.0, 0.0, 1.5, 0.0], # Latent 3
            [0.0, 0.0, 0.0, 4.8], [0.0, 0.0, 0.0, 2.7], [0.0, 0.0, 0.0, 1.4]  # Latent 4
        ])
    else:
        raise ValueError(f"Invalid scenario '{scenario}' for Lambda specification.")
    
    # 12 items x 2 covariates
    beta = np.array([
        [0.3,  0.5], [0.1,  0.2], [-0.1, 0.2], [0.2, -0.1], [0.1,  0.3],  # Items 1-5
        [-0.1,-0.2], [-0.2,-0.1], [0.4,  0.1], [0.2, -0.4], [-0.3, 0.2],  # Items 6-10
        [0.1, -0.1], [0.2,  0.3]                                          # Items 11-12
    ])
    
    # Thresholds: Items 1-5 are binary, Items 6-12 are ordinal (4 categories)
    theta = [
        np.array([2.3]), np.array([1.5]), np.array([0.8]), np.array([1.2]), np.array([2.1]), # 1-5 (Binary)
        np.array([-4.0, -1.0, 2.7]), np.array([-5.5, -2.5, 2.6]), # 6-7 (Ordinal)
        np.array([-4.5, -1.5, 2.0]), np.array([-3.0,  0.0, 3.0]), # 8-9 (Ordinal)
        np.array([-6.0, -2.0, 1.5]), np.array([-5.0, -1.0, 2.5]), # 10-11 (Ordinal)
        np.array([-4.0, -0.5, 2.0])                               # 12 (Ordinal)
    ]
    
    # Residual variances for 12 items
    sigma_b = np.array([3.7, 3.4, 4.8, 3.1, 4.0, 6.1, 5.1, 1.7, 2.5, 3.3, 4.2, 2.8])
    
    # Latent OU Process Parameters (q=4)
    if 'G1' in scenario or 'G2' in scenario:
        # 1. Base Drift (Directly decomposable into SPD + Skew)
        S_base = np.diag([0.2, 0.3, 0.2, 0.4])
        A_base = np.array([
            [ 0.0,  0.8,  0.0,  0.0],
            [-0.8,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.6],
            [ 0.0,  0.0, -0.6,  0.0]
        ])
        Gamma_base = S_base + A_base
        
        if 'G1' in scenario: 
            # Scenario A: No transformation. Symmetric part is Positive Definite.
            Gamma_raw = Gamma_base
            
        elif 'G2' in scenario: 
            # Scenario B: Apply strong shear transformation. 
            # Gamma remains stable, but its symmetric part is NO LONGER Positive Definite.
            P = np.array([
                [ 1.0,  2.5,  0.0, -1.5],
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  2.0],
                [ 0.0,  0.0,  0.0,  1.0]
            ])
            P_inv = np.linalg.inv(P)
            Gamma_raw = P @ Gamma_base @ P_inv 

        # 2. Generate physically consistent stationary covariance (Omega)
        # We assume independent unit system noise (Q)
        Q_raw = np.eye(4)
        
        # Solve the Lyapunov equation: Gamma * Omega + Omega * Gamma^T = Q
        Omega_raw = solve_continuous_lyapunov(Gamma_raw, Q_raw)
        
        # 3. Standardize to a Correlation Matrix (1s on diagonal) for identifiability
        std_devs = np.sqrt(np.diag(Omega_raw))
        D_inv_sqrt = np.diag(1.0 / std_devs)
        D_sqrt = np.diag(std_devs)
        
        # Scale Omega to have 1s on the diagonal
        Omega = D_inv_sqrt @ Omega_raw @ D_inv_sqrt
        
        # Apply the matching similarity transformation to Gamma to preserve Lyapunov dynamics
        Gamma = D_inv_sqrt @ Gamma_raw @ D_sqrt

        # Extract the 6 upper-triangle correlations
        rows, cols = np.triu_indices(4, k=1)
        rho_vec = Omega[rows, cols]

    elif 'G3' in scenario:
        # [corr12, corr13, corr14, corr23, corr24, corr34]
        rho_vec = np.array([0.6, 0.2, -0.7, 0.1, -0.5, -0.1])
        
        # 1. Initialize a 4x4 identity matrix (1s on the diagonal)
        Omega = np.eye(4)
        
        # 2. Get the indices for the upper triangle (k=1 excludes the diagonal)
        rows, cols = np.triu_indices(4, k=1)
        
        # 3. Assign the rho vector to the upper triangle
        Omega[rows, cols] = rho_vec
        
        # 4. Mirror the upper triangle to the lower triangle to make it symmetric
        Omega[cols, rows] = rho_vec
        
        # --- CRITICAL CHECK ---
        # Not all combinations of correlations result in a valid correlation matrix.
        # We must check if the matrix is positive-definite.
        eigenvalues = np.linalg.eigvals(Omega)
        if not np.all(eigenvalues > 0):
            raise ValueError(
                f"Invalid correlation matrix! The eigenvalues are {eigenvalues}. "
                "The chosen rho_vec does not form a positive-definite matrix."
            )
        
        # S: Controls the speed of mean-reversion (must be positive definite)
        S = 0.2 * np.eye(4) 
        
        # A: Skew-symmetric matrix introducing rotations (complex eigenvalues)
        # Here, we link latent factors 1 & 2 to rotate together, and 3 & 4 to rotate together
        A = np.array([
            [ 0.0,  0.6,  0.0,  0.0],
            [-0.6,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.4],
            [ 0.0,  0.0, -0.4,  0.0]
        ])
        
        # Calculate Gamma: guaranteed to be positive stable and valid with Omega!
        Gamma = (S + A) @ np.linalg.inv(Omega)
    else:
        raise ValueError(f"Invalid scenario '{scenario}' for Gamma/Omega specification.")
    
    c_latent = np.array([0.5, -0.3, 0.2, -0.4])
    A_latent = np.array([
        [ 0.4, -0.2],
        [-0.3,  0.5],
        [ 0.2,  0.1],
        [-0.1, -0.3]
    ])
        
    # Missing Data Parameters (MAR) - extended for 12 items
    K_miss = np.array([
        [-0.25, -0.27, -0.6, -1.8], [-0.24,  0.12, -0.4, -1.8],
        [-0.27, -0.11, -0.7, -1.5], [-0.22, -0.20, -0.7, -1.7],
        [ 0.10, -0.10, -0.5, -1.6], [ 0.25, -0.16, -0.7, -1.8], 
        [ 0.20, -0.13, -0.8, -1.9], [ 0.25,  0.06, -1.1, -1.5],
        [-0.15,  0.05, -0.6, -1.4], [ 0.18, -0.08, -0.9, -1.7], 
        [-0.20,  0.15, -0.5, -1.5], [ 0.22, -0.11, -0.8, -1.8]  
    ])
    
    n_i_choices = np.arange(2, 13)
    n_i_weights = np.array([10, 25, 22, 18, 10, 5, 5, 2, 1.5, 0.8, 0.7]) / 100.0
    
    dataset = []
    
    for i in range(N):
        n_i = np.random.choice(n_i_choices, p=n_i_weights)
        b_i = np.random.normal(0, sigma_b)
        
        # Time-invariant covariates Z_i
        Z_i = np.array([np.random.binomial(1, 0.5), np.random.normal(0, 1)])
            
        d_ij = np.random.uniform(0.5, 1.5, size=n_i)
        d_ij[0] = 0.0  
        t_ij = np.cumsum(d_ij)
        
        xi_star_history = np.zeros((n_i, 4)) 
        xi_i = np.zeros((n_i, 4))            
        Y_i = np.zeros((n_i, 12))
        Y_i_missing = np.zeros((n_i, 12))
        
        for j in range(n_i):
            x_ij = np.array([np.random.binomial(1, 0.5), np.random.normal(0, 1)])
            
            if j == 0:
                xi_star = np.random.multivariate_normal([0, 0, 0, 0], Omega)
            else:
                delta_t = d_ij[j]
                transition_mat = expm(-Gamma * delta_t)
                mean_xi = transition_mat @ xi_star_history[j-1]
                cov_xi = Omega - (transition_mat @ Omega @ transition_mat.T)
                xi_star = np.random.multivariate_normal(mean_xi, cov_xi)
                
            xi_star_history[j] = xi_star
            
            # Apply Mean Shifts based on Scenario
            if 'C1' in scenario:
                trend_offset = np.zeros(4)
            elif 'C2' in scenario:
                trend_offset = c_latent * t_ij[j]
            elif 'C3' in scenario:
                trend_offset = (A_latent @ Z_i) * t_ij[j]
            elif 'C4' in scenario:
                trend_offset = (A_latent @ Z_i + c_latent) * t_ij[j]
            else:
                raise ValueError(f"Invalid scenario '{scenario}' for mean structure specification.")
                
            xi_i[j] = xi_star + trend_offset
            
            for k in range(12):
                linear_predictor = np.dot(beta[k], x_ij) - np.dot(Lambda[k], xi_i[j]) + b_i[k]
                probs_le_m = expit(theta[k] + linear_predictor)
                
                probs = np.zeros(len(theta[k]) + 1)
                probs[0] = probs_le_m[0]
                for m in range(1, len(theta[k])):
                    probs[m] = probs_le_m[m] - probs_le_m[m-1]
                probs[-1] = 1.0 - probs_le_m[-1]
                
                probs = np.clip(probs, 0, 1)
                probs /= probs.sum()
                Y_i[j, k] = np.random.choice(len(probs), p=probs)
                
            if j > 0:
                for k in range(12):
                    logit_p_miss = (K_miss[k, 0] + K_miss[k, 1] * x_ij[0] + 
                                    K_miss[k, 2] * x_ij[1] + K_miss[k, 3] * Y_i[j-1, k])
                    p_miss = expit(logit_p_miss)
                    Y_i_missing[j, k] = np.nan if np.random.rand() < p_miss else Y_i[j, k]
            else:
                Y_i_missing[j, :] = Y_i[j, :]
                
            dataset.append({
                'id': i + 1, 'time': t_ij[j],
                'x1': x_ij[0], 'x2': x_ij[1],
                'Z1': Z_i[0], 'Z2': Z_i[1],
                'xi1': xi_i[j, 0], 'xi2': xi_i[j, 1], 'xi3': xi_i[j, 2], 'xi4': xi_i[j, 3],
                'Y1': Y_i_missing[j, 0], 'Y2': Y_i_missing[j, 1], 'Y3': Y_i_missing[j, 2],
                'Y4': Y_i_missing[j, 3], 'Y5': Y_i_missing[j, 4], 'Y6': Y_i_missing[j, 5],
                'Y7': Y_i_missing[j, 6], 'Y8': Y_i_missing[j, 7], 'Y9': Y_i_missing[j, 8],
                'Y10': Y_i_missing[j, 9], 'Y11': Y_i_missing[j, 10], 'Y12': Y_i_missing[j, 11]
            })
            
    return dataset

# ==========================================
# 2. General Ground Truth & Data Prep
# ==========================================
def add_param_to_dict(d, name, val):
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            for i, v in enumerate(val):
                d[f"{name}[{i+1}]"] = float(v)
        elif val.ndim == 2:
            for i in range(val.shape[0]):
                for j in range(val.shape[1]):
                    d[f"{name}[{i+1},{j+1}]"] = float(val[i, j])
    else:
        d[name] = float(val)

def create_ground_truth_dict(scenario='L2G3C4'):
    truths = {}
    
    # 5 Binary items
    truths['theta1'] = 2.3
    truths['theta2'] = 1.5
    truths['theta3'] = 0.8
    truths['theta4'] = 1.2
    truths['theta5'] = 2.1
    
    # 7 Ordinal items
    add_param_to_dict(truths, 'theta6', np.array([-4.0, -1.0, 2.7]))
    add_param_to_dict(truths, 'theta7', np.array([-5.5, -2.5, 2.6]))
    add_param_to_dict(truths, 'theta8', np.array([-4.5, -1.5, 2.0]))
    add_param_to_dict(truths, 'theta9', np.array([-3.0,  0.0, 3.0]))
    add_param_to_dict(truths, 'theta10', np.array([-6.0, -2.0, 1.5]))
    add_param_to_dict(truths, 'theta11', np.array([-5.0, -1.0, 2.5]))
    add_param_to_dict(truths, 'theta12', np.array([-4.0, -0.5, 2.0]))
    
    # Extract the primary (non-zero) loading for each item based on L condition
    if 'L1' in scenario:
        add_param_to_dict(truths, 'lambda', np.array([0.8, 1.2, 1.1, 0.9, 1.4, 1.0, 0.7, 1.1, 0.9, 1.2, 0.8, 1.0]))
    elif 'L2' in scenario:
        add_param_to_dict(truths, 'lambda', np.array([1.2, 4.0, 4.1, 3.1, 5.2, 3.0, 1.7, 2.4, 1.5, 4.8, 2.7, 1.4]))
    else:
        raise ValueError(f"Invalid scenario '{scenario}' for Lambda ground truth.")
    
    # Beta and Sigma_bk (Shared across all scenarios)
    add_param_to_dict(truths, 'beta', np.array([
        [0.3,  0.5], [0.1,  0.2], [-0.1, 0.2], [0.2, -0.1], [0.1,  0.3], 
        [-0.1,-0.2], [-0.2,-0.1], [0.4,  0.1], [0.2, -0.4], [-0.3, 0.2], 
        [0.1, -0.1], [0.2,  0.3]
    ]))
    
    add_param_to_dict(truths, 'sigma_bk', np.array([3.7, 3.4, 4.8, 3.1, 4.0, 6.1, 5.1, 1.7, 2.5, 3.3, 4.2, 2.8]))
    
    # Latent OU Process Parameters (Gamma and Rho)
    if 'G1' in scenario or 'G2' in scenario:
        S_base = np.diag([0.2, 0.3, 0.2, 0.4])
        A_base = np.array([
            [ 0.0,  0.8,  0.0,  0.0],
            [-0.8,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.6],
            [ 0.0,  0.0, -0.6,  0.0]
        ])
        Gamma_base = S_base + A_base
        
        if 'G1' in scenario: 
            Gamma_raw = Gamma_base
        elif 'G2' in scenario: 
            P = np.array([
                [ 1.0,  2.5,  0.0, -1.5],
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  2.0],
                [ 0.0,  0.0,  0.0,  1.0]
            ])
            P_inv = np.linalg.inv(P)
            Gamma_raw = P @ Gamma_base @ P_inv 

        Q_raw = np.eye(4)
        Omega_raw = solve_continuous_lyapunov(Gamma_raw, Q_raw)
        
        std_devs = np.sqrt(np.diag(Omega_raw))
        D_inv_sqrt = np.diag(1.0 / std_devs)
        D_sqrt = np.diag(std_devs)
        
        Omega = D_inv_sqrt @ Omega_raw @ D_inv_sqrt
        Gamma = D_inv_sqrt @ Gamma_raw @ D_sqrt

        rows, cols = np.triu_indices(4, k=1)
        rho_vec = Omega[rows, cols]
        
    elif 'G3' in scenario:
        rho_vec = np.array([0.6, 0.2, -0.7, 0.1, -0.5, -0.1])
        Omega = np.eye(4)
        rows, cols = np.triu_indices(4, k=1)
        Omega[rows, cols] = rho_vec
        Omega[cols, rows] = rho_vec
        
        S = 0.2 * np.eye(4) 
        A = np.array([
            [ 0.0,  0.6,  0.0,  0.0],
            [-0.6,  0.0,  0.0,  0.0],
            [ 0.0,  0.0,  0.0,  0.4],
            [ 0.0,  0.0, -0.4,  0.0]
        ])
        Gamma = (S + A) @ np.linalg.inv(Omega)
    else:
        raise ValueError(f"Invalid scenario '{scenario}' for Gamma/rho ground truth.")
        
    add_param_to_dict(truths, 'Gamma', Gamma)
    add_param_to_dict(truths, 'rho', rho_vec)
        
    # Latent Mean Shifts (c_latent and A_latent) based on C condition
    if 'C2' in scenario or 'C4' in scenario:
        add_param_to_dict(truths, 'c_latent', np.array([0.5, -0.3, 0.2, -0.4]))
        
    if 'C3' in scenario or 'C4' in scenario:
        add_param_to_dict(truths, 'A_latent', np.array([
            [ 0.4, -0.2],
            [-0.3,  0.5],
            [ 0.2,  0.1],
            [-0.1, -0.3]
        ]))
            
    return truths

def prepare_stan_data(dataset):
    df = pd.DataFrame(dataset)
    df['deltat'] = df.groupby('id')['time'].diff().fillna(0.0)
    repme = df.groupby('id').size().values
    cumu = np.cumsum(repme)
    
    Y_raw = df[['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12']].values
    missing_ID = np.isnan(Y_raw).astype(int)
    Y = np.nan_to_num(Y_raw, nan=-99).astype(int)
    
    # Ordinal items are index 5 through 11 (items 6-12)
    # Increment by 1 for 1-based indexing in Stan's ordered logistic
    for col_idx in range(5, 12):
        valid_mask = (missing_ID[:, col_idx] == 0)
        Y[valid_mask, col_idx] += 1
        
    stan_data = {
        'N': len(df), 'Nsub': df['id'].nunique(), 'K': 12, 'R': 4, 'p': 2, 'q': 2,
        'ID': df['id'].values.astype(int), 'cumu': cumu.astype(int), 'repme': repme.astype(int),
        'Y': Y, 'missing_ID': missing_ID, 'deltat': df['deltat'].values, 
        'time': df['time'].values,
        'X': df[['x1', 'x2']].values,
        'Z': df[['Z1', 'Z2']].fillna(0.0).values, 
        'ncate6': 4, 'ncate7': 4, 'ncate8': 4, 'ncate9': 4, 
        'ncate10': 4, 'ncate11': 4, 'ncate12': 4
    }
    return stan_data

# ==========================================
# 3. Execution & Aggregation Pipeline
# ==========================================
def evaluate_model_performance(stan_file_path, dataset, run_id, scenario='L2G3C4', iter_sampling=1000, iter_warmup=1000, chains=3):
    stan_data = prepare_stan_data(dataset)
    ground_truths = create_ground_truth_dict(scenario)
    
    # HPC BULLETPROOFING: Explicitly point to the pre-compiled executable
    exe_path = stan_file_path.replace('.stan', '')
    model_dir = "./compiled_models"
    os.makedirs(model_dir, exist_ok=True)
    exe_path = os.path.join(model_dir, os.path.basename(exe_path))
    if os.path.exists(exe_path):
        model = cmdstanpy.CmdStanModel(exe_file=exe_path)
    else:
        raise FileNotFoundError(f"Compiled executable not found at {exe_path}. Please ensure the model is pre-compiled and the path is correct.")

    start_time = time.time()
    fit = model.sample(
        data=stan_data, iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        chains=chains, parallel_chains=chains, adapt_delta=0.95, max_treedepth=12,
        show_progress=False 
    )
    run_time = time.time() - start_time

    output_dir = "../raw_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- DEFENSIVE DIAGNOSTICS EXTRACTION ---
    try:
        divergences = int(fit.divergences.sum())
    except AttributeError:
        divergences = 0 
        
    try:
        treedepths = int(np.sum(fit.method_variables()['treedepth__'] >= 12))
        energies = fit.method_variables()['energy__'] 
        ebfmis = []
        for c in range(energies.shape[1]):
            chain_energy = energies[:, c]
            numer = np.sum(np.diff(chain_energy)**2)
            denom = np.sum((chain_energy - np.mean(chain_energy))**2)
            ebfmis.append(numer / denom)
        mean_ebfmi = np.mean(ebfmis)
    except (KeyError, AttributeError):
        treedepths = 0
        mean_ebfmi = np.nan

    try:
        summary_df = fit.summary(percentiles=[2.5, 97.5])
                            
    except Exception as e:
        print(f"Error in fit.summary(): {e}. Attempting ArviZ fallback...")
        
        # Convert CmdStanPy output directly to an ArviZ object in memory
        idata = az.from_cmdstanpy(fit)
        
        # Generate summary in Python (bypassing the C++ stansummary binary)
        summary_df = az.summary(idata, hdi_prob=0.95)
        
        # Rename columns to perfectly match your existing downstream logic
        summary_df = summary_df.rename(columns={
            'mean': 'Mean', 
            'hdi_2.5%': '2.5%', 
            'hdi_97.5%': '97.5%', 
            'r_hat': 'R_hat',
            'mcse_mean': 'MCSE',
            'ess_bulk': 'ESS_bulk'
        })
        
        # ArviZ outputs 0-based indices (e.g., theta[0]). 
        # This shifts it back to 1-based indexing (theta[1]) to match your ground truths
        summary_df.index = [re.sub(r'\[(\d+)\]', lambda m: f"[{int(m.group(1))+1}]", idx) 
                            if '[' in idx else idx 
                            for idx in summary_df.index]
            
    cols = summary_df.columns.tolist()
    
    lower_col = '2.5%' if '2.5%' in cols else ('5%' if '5%' in cols else None)
    upper_col = '97.5%' if '97.5%' in cols else ('95%' if '95%' in cols else None)
    rhat_col = 'R_hat' if 'R_hat' in cols else ('Rhat' if 'Rhat' in cols else None)
    mcse_col = 'MCSE' if 'MCSE' in cols else None

    # Get the complete union of all parameters (shared, truth-only, model-only)
    truth_params = set(ground_truths.keys())
    model_params = set(summary_df.index)
    all_params = truth_params.union(model_params)

    # DROP STAN INTERNALS: Remove lp__ or any other parameter ending in a double underscore
    all_params = {p for p in all_params if not p.endswith('__')}

    results = []
    
    for param_name in all_params:
        # 1. Fetch ground truth if it exists, otherwise NA
        true_val = ground_truths.get(param_name, np.nan)
        
        # 2. Fetch model estimates if they exist, otherwise NA
        if param_name in model_params:
            est_mean = summary_df.loc[param_name, 'Mean']
            lower_ci = summary_df.loc[param_name, lower_col] if lower_col else np.nan
            upper_ci = summary_df.loc[param_name, upper_col] if upper_col else np.nan
            rhat = summary_df.loc[param_name, rhat_col] if rhat_col else np.nan
            mcse = summary_df.loc[param_name, mcse_col] if mcse_col else np.nan
            
            if 'ESS_bulk' in cols: ess = summary_df.loc[param_name, 'ESS_bulk']
            elif 'N_Eff' in cols: ess = summary_df.loc[param_name, 'N_Eff']
            else: ess = np.nan
                
            ess_sec = ess / run_time if run_time > 0 and pd.notna(ess) else np.nan
        else:
            est_mean = lower_ci = upper_ci = rhat = mcse = ess = ess_sec = np.nan
        
        # 3. Calculate comparative metrics ONLY if both truth and estimate exist
        if pd.notna(true_val) and pd.notna(est_mean):
            rbias = ((est_mean - true_val) / true_val) if true_val != 0 else np.nan
            mse = (est_mean - true_val) ** 2
            coverage = 1 if (lower_ci <= true_val <= upper_ci) else 0
        else:
            rbias = mse = coverage = np.nan
        
        results.append({
            'Run_ID': run_id, 'Parameter': param_name, 'True_Value': true_val, 'Estimate': est_mean,
            '2.5%': lower_ci, '97.5%': upper_ci, 'R_hat': rhat, 'ESS': ess,
            'MCSE': mcse, 'ESS_sec': ess_sec, 'Rbias': rbias, 'MSE': mse, 'Coverage': coverage,
            'Time_s': run_time, 'Divergences': divergences, 'Max_Treedepths': treedepths, 'E_BFMI': mean_ebfmi
        })
            
    metrics_df = pd.DataFrame(results)
    model_name = os.path.splitext(os.path.basename(stan_file_path))[0]
    metrics_df.to_csv(os.path.join(output_dir, f"results_{scenario}_{model_name}_run{run_id}.csv"), index=False)
    
    return True

def generate_simulation_table(scenario, model_name):
    print(f"\nAggregating results across all runs for {scenario} - {model_name}...")
    source_dir = os.path.join("..", "corrected_results")
    target_dir = os.path.join("..", "summarized_results")
    os.makedirs(target_dir, exist_ok=True)
    
    file_pattern = os.path.join(source_dir, f"results_{scenario}_{model_name}_run*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("No simulation files found to aggregate.")
        return None
        
    df_list = []
    total_considered = len(all_files)
    total_aggregated = 0
    total_discarded_errors = 0
    
    for f in all_files:
        try:
            df = pd.read_csv(f)
            if (df['R_hat'] < 1.1).all():
                df_list.append(df)
                total_aggregated += 1
                
        except KeyError as e:
            print(f"  [Warning] Discarding {f}: Missing expected column {e}")
            total_discarded_errors += 1
        except pd.errors.EmptyDataError:
            print(f"  [Warning] Discarding {f}: File is completely empty.")
            total_discarded_errors += 1
        except Exception as e:
            print(f"  [Warning] Discarding {f}: Unexpected error -> {e}")
            total_discarded_errors += 1
            
    print(f"\n--- Summary ---")
    print(f"Total tables considered: {total_considered}")
    if total_discarded_errors > 0:
        print(f"Total tables discarded due to errors/missing data: {total_discarded_errors}")
    print(f"Total tables aggregated (R_hat < 1.05): {total_aggregated}")
    print(f"----------------\n")
    
    if not df_list:
        print("No tables met the criteria for aggregation. Aborting.")
        return None

    combined_df = pd.concat(df_list, ignore_index=True)
    
    run_level_df = combined_df.groupby('Run_ID').agg(
        Time=('Time_s', 'first'),
        Divergences=('Divergences', 'first'),
        Treedepths=('Max_Treedepths', 'first'),
        E_BFMI=('E_BFMI', 'first')
    )
    
    print("\n" + "="*50)
    print(" 🛠️ RUN-LEVEL HMC DIAGNOSTICS (AVERAGE) 🛠️ ")
    print("="*50)
    print(f"Total Wall-Clock Time (s): {run_level_df['Time'].mean():.2f}")
    print(f"Divergent Transitions:     {run_level_df['Divergences'].mean():.2f}")
    print(f"Max Treedepth Hits:        {run_level_df['Treedepths'].mean():.2f}")
    print(f"E-BFMI:                    {run_level_df['E_BFMI'].mean():.3f}")
    print("="*50 + "\n")
    
    table_df = combined_df.groupby('Parameter').agg(
        True_Value=('True_Value', 'first'),
        RB=('Rbias', 'mean'),             
        MSE=('MSE', 'mean'),              
        CP=('Coverage', lambda x: x.mean() * 100), 
        ESS=('ESS', 'mean'),
        ESS_sec=('ESS_sec', 'mean'),
        MCSE=('MCSE', 'mean'),
        Rhat=('R_hat', 'max')             
    ).reset_index()
    
    table_df['RB'] = table_df['RB'].round(3)
    table_df['MSE'] = table_df['MSE'].round(3)
    table_df['CP'] = table_df['CP'].round(1)
    table_df['ESS'] = table_df['ESS'].round(1)
    table_df['ESS_sec'] = table_df['ESS_sec'].round(3)
    table_df['MCSE'] = table_df['MCSE'].round(4)
    table_df['Rhat'] = table_df['Rhat'].round(3)
    
    final_filename = f"TABLE_{scenario}_{model_name}.csv"
    final_path = os.path.join(target_dir, final_filename)
    table_df.to_csv(final_path, index=False)
    print(f"Success! Final aggregate parameter table saved to: {final_filename}")
    
    return table_df

def aggregate_cross_model_results(scenario, model_names):
    print(f"\nMerging results across models for Scenario: {scenario}...")
    source_dir = os.path.join("..", "summarized_results")
    target_dir = os.path.join("..", "aggregated_results")
    os.makedirs(target_dir, exist_ok=True)
    
    dataframes = []
    for model in model_names:
        filename = f"TABLE_{scenario}_{model}.csv"
        filepath = os.path.join(source_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  [Warning] Missing file: {filename} in {source_dir}. Skipping this model.")
            continue
            
        df = pd.read_csv(filepath)
        rename_map = {
            col: f"{col}_{model}" 
            for col in df.columns 
            if col not in ['Parameter', 'True_Value']
        }
        df.rename(columns=rename_map, inplace=True)
        dataframes.append(df)
        
    if not dataframes:
        print("No model tables found to merge.")
        return None
        
    final_combined_df = reduce(
        lambda left, right: pd.merge(left, right, on=['Parameter', 'True_Value'], how='outer'), 
        dataframes
    )
    
    base_metrics = ['RB', 'MSE', 'CP', 'ESS', 'ESS_sec', 'MCSE','Rhat']
    ordered_cols = ['Parameter', 'True_Value']
    for metric in base_metrics:
        for model in model_names:
            col_name = f"{metric}_{model}"
            if col_name in final_combined_df.columns:
                ordered_cols.append(col_name)
                
    remaining_cols = [c for c in final_combined_df.columns if c not in ordered_cols]
    final_combined_df = final_combined_df[ordered_cols + remaining_cols]
    
    output_filename = f"FINAL_COMPARISON_{scenario}.csv"
    output_filepath = os.path.join(target_dir, output_filename)
    final_combined_df.to_csv(output_filepath, index=False)
    
    print(f"Success! Cross-model comparison saved to: {output_filepath}")
    return final_combined_df

# ==========================================
# 4. Multiprocessing Wrapper & Main Execution
# ==========================================
def parallel_worker(args_dict):
    run_id = args_dict['run_id']
    print(f"  -> Starting Run {run_id}...")
    try:
        dataset = generate_lou_simulation_data(
            N=600, 
            scenario=args_dict['scenario'], 
            random_state=42 + run_id
        )
        
        evaluate_model_performance(
            stan_file_path=args_dict['stan_file'], 
            dataset=dataset,
            run_id=run_id,
            scenario=args_dict['scenario'],
            iter_warmup=args_dict['warmup'],
            iter_sampling=args_dict['sampling'],
            chains=args_dict['chains']
        )
        print(f"  ✅ Completed Run {run_id}")
        return run_id, True
    except Exception as e:
        print(f"  ❌ Failed Run {run_id}: {str(e)}")
        return run_id, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOU Simulation on HPC")
    parser.add_argument("--scenario", type=str, required=True, help="Data generation scenario")
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Path(s) to the Stan model file(s), separated by spaces")
    parser.add_argument("--chains", type=int, default=3, help="Number of MCMC chains")
    parser.add_argument("--warmup", type=int, default=2500, help="Warmup iterations")
    parser.add_argument("--sampling", type=int, default=2500, help="Sampling iterations")
    
    parser.add_argument("--start_run", type=int, default=1, help="Starting run ID")
    parser.add_argument("--end_run", type=int, default=200, help="Ending run ID")
    parser.add_argument("--aggregate_only", action="store_true", help="Only run the table aggregation")
    parser.add_argument("--compile_only", action="store_true", help="Only compile the model, do not run sims") 
    parser.add_argument("--cross_aggregate", action="store_true", help="Merge multiple model tables into one final comparison table")
    parser.add_argument("--workers", type=int, default=None, help="Manual override for parallel workers")
    
    args = parser.parse_args()
    scenario_to_run = args.scenario
    model_names = [os.path.splitext(os.path.basename(m))[0] for m in args.models]
    stan_file = args.models[0]
    model_name = model_names[0] 
    
    if args.cross_aggregate:
        aggregate_cross_model_results(scenario=scenario_to_run, model_names=model_names)
        exit(0)

    if args.compile_only:
        print(f"Pre-compiling Stan Model: {model_name}...")
        model_dir = "./models"
        compiled_dir = "./compiled_models"

        os.makedirs(compiled_dir, exist_ok=True)
        if not os.path.exists(model_dir):
            print(f"Error: Models directory '{model_dir}' not found. Please ensure the path is correct.")
            exit(1)
        stan_file_path = os.path.join(model_dir, stan_file)
        # name of compiled binary (no .stan extension)
        exe_path = os.path.join(compiled_dir, model_name)

        _ = cmdstanpy.CmdStanModel(
            stan_file=stan_file_path,
            exe_file=exe_path
        )
        print("Compilation successful. Exiting.")
        exit(0)

    if args.aggregate_only:
        generate_simulation_table(scenario=scenario_to_run, model_name=model_name)
        exit(0)

    try:
        available_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    except TypeError:
        available_cores = os.cpu_count()
        
    if args.workers is None:
        safe_cores = max(1, available_cores - 2)
        optimal_workers = safe_cores // args.chains
        args.workers = max(1, optimal_workers)
    
    print(f"Starting HPC Monte Carlo Simulation: Scenario = {scenario_to_run}, Model = {model_name}")
    print(f"Running Sims {args.start_run} to {args.end_run} | Total Cores Detected: {available_cores}")
    print(f"Running with {args.workers} Parallel Workers ({args.chains} chains each)\n")
    
    tasks = []
    for run_id in range(args.start_run, args.end_run + 1):
        tasks.append({
            'run_id': run_id,
            'scenario': scenario_to_run,
            'stan_file': stan_file,
            'warmup': args.warmup,
            'sampling': args.sampling,
            'chains': args.chains
        })

    successful_runs = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(parallel_worker, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            run_id, success = future.result()
            if success:
                successful_runs += 1

    print(f"\nNode finished its chunk. ({successful_runs}/{(args.end_run - args.start_run) + 1} successful)")