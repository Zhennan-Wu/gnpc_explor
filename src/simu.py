import torch
import jax
import jax.numpy as jnp
import numpy as np
from dfa_ou import OUDynamicFactorModel as DFM_V1
from dfa_ou_autograd import OUDynamicFactorModel as DFM_V2
from dfa_ou_damp import OUDynamicFactorModel as DFM_V3
from nmf_ode_l1reg import NMF_LinearODE_Model as DFM_ODE
from lou_test import OULatentModel
from comp_utils import NumPyroModelWrapper, bridge_to_jax
from visual import ModelVisualizer


def generate_data(D, K, T, N_subjects, noise_std=0.3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Lambda = torch.abs(torch.randn(D, K, device=device) * 0.5)
    rho, drift_base, gamma = torch.rand(K, device=device) * 0.2 + 0.05, torch.randn(K, device=device) * 0.5, torch.eye(K, device=device) * 0.05 
    all_data, all_times, all_covs, all_factors = [], [], [], []
    for i in range(N_subjects):
        covs, times = torch.randn(3, device=device), torch.linspace(0, 10, T, device=device)
        factors, f_curr = torch.zeros(T, K, device=device), torch.randn(K, device=device) * 0.5 
        for t in range(T):
            if t > 0: f_curr = f_curr + rho * ((drift_base + 0.3*covs[2]) - f_curr) * (times[t]-times[t-1]) + torch.randn(K, device=device) * 0.02
            factors[t] = f_curr
        all_data.append(factors @ Lambda.T + torch.randn(T, D, device=device) * noise_std)
        all_times.append(times); all_covs.append(covs); all_factors.append(factors)
    gt_params = {'Lambda': Lambda, 'rho': rho, 'Gamma': gamma, 'alpha': drift_base, 'sigma_obs': torch.tensor(noise_std, device=device), 'A': -torch.diag(rho), 'b': drift_base * rho, 'Phi': torch.zeros(K, 3, device=device)}
    return all_data, all_times, all_covs, gt_params, all_factors


def generate_easy_data(D, K, T, N_subjects, process_noise=0.1, obs_noise=0.5, non_negative=False):
    """
    Improved Generator:
    - process_noise=0: Deterministic Case
    - obs_noise=0: Noise-free observations
    - non_negative=True: NMF ground truth
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Generate Loadings
    Lambda = torch.randn(D, K, device=device)
    if non_negative:
        Lambda = torch.abs(Lambda)
    mask = torch.rand(D, K, device=device) > 0.5 # Sparsity
    Lambda = Lambda * mask.float()
    
    rho = torch.rand(K, device=device) * 0.5 + 0.1
    all_data, all_times, all_covs = [], [], []
    
    for _ in range(N_subjects):
        times = torch.cumsum(torch.rand(T, device=device) * 0.5 + 0.1, dim=0)
        factors = torch.zeros(T, K, device=device)
        f_curr = torch.randn(K, device=device)
        if non_negative: f_curr = torch.abs(f_curr)
        
        for t in range(T):
            if t > 0:
                dt = times[t] - times[t-1]
                # If process_noise is 0, this is a pure ODE
                noise = torch.randn(K, device=device) * process_noise
                f_curr = torch.exp(-rho * dt) * f_curr + noise
            if non_negative: f_curr = torch.clamp(f_curr, min=0)
            factors[t] = f_curr
            
        # Observation
        noise_y = torch.randn(T, D, device=device) * obs_noise
        obs = factors @ Lambda.T + noise_y
        
        all_data.append(obs)
        all_times.append(times)
        all_covs.append(torch.randn(3, device=device)) 

    return all_data, all_times, all_covs, Lambda


def generate_disease_proteomics_data(D, K, T, N_subjects, noise_std=0.3, sparsity=0.8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. STRUCTURED LOADINGS (Lambda)
    # Proteins belong to specific functional modules (blocks)
    Lambda = torch.zeros(D, K, device=device)
    block_size = D // K
    for k in range(K):
        # Create overlapping blocks to test factor deconvolution
        idx = torch.arange(k * block_size, min((k + 2) * block_size, D))
        Lambda[idx, k] = torch.randn(len(idx), device=device) * 1.5
    
    # 2. GLOBAL DYNAMICS (Mean Reversion & Progression Goal)
    rho = torch.rand(K, device=device) * 0.2 + 0.05 
    drift_base = torch.randn(K, device=device) * 0.5 
    
    all_data, all_times, all_covs, all_factors = [], [], [], []
    
    for i in range(N_subjects):
        # Generate 3 Covariates: e.g., [Age (cont), Sex (bin), Treatment (bin)]
        # We'll make these covariates actually affect the drift (progression)
        covs = torch.tensor([torch.randn(1), 
                             torch.randint(0, 2, (1,)), 
                             torch.randint(0, 2, (1,))], device=device).flatten().float()
        
        # Subject-specific drift: Baseline drift + effect of covariates
        # This tests if the model can attribute progression to external metadata
        subject_drift = drift_base + (0.3 * covs[2]) # Treatment reduces/increases drift
        
        times = torch.cumsum(torch.exp(torch.randn(T, device=device) * 0.2 - 1.0), dim=0)
        subject_intercept = torch.randn(D, device=device) * 0.5 + (covs[0] * 0.1) # Age offset
        
        factors = torch.zeros(T, K, device=device)
        f_curr = torch.randn(K, device=device) * 0.5 
        
        for t in range(T):
            if t > 0:
                dt = times[t] - times[t-1]
                # OU Process: df = rho(drift - f)dt + sigma*dW
                f_curr = f_curr + rho * (subject_drift - f_curr) * dt + torch.randn(K, device=device) * 0.05
            factors[t] = f_curr
            
        # Observation with Heteroscedastic Noise
        clean_signal = subject_intercept + factors @ Lambda.T
        noise = torch.randn(T, D, device=device) * noise_std * (1 + 0.05 * clean_signal.abs())
        obs = clean_signal + noise
        
        all_data.append(obs)
        all_times.append(times)
        all_covs.append(covs)
        all_factors.append(factors)

    return all_data, all_times, all_covs, Lambda, drift_base


def run_benchmark(D=10, K=2, T=5, subjects=10, n_runs=2, epochs=20):
    data, times, covs, gt_params, factors = generate_data(D, K, T, subjects)
    gt_data = {'traj': factors, 'times': times, 'data': data, 'covs': covs}
    viz = ModelVisualizer(gt_params, gt_data)
    # trained_models = {name: [] for name in ["V1_EM", "V2_Autograd", "V3_Robust"]}
    # trained_models = {name: [] for name in ["NMF_ODE", "LOU", "V3_Robust"]}
    trained_models = {name: [] for name in ["NMF_ODE"]}
    for r in range(n_runs):
        print(f"--- Run {r+1}/{n_runs} ---")
        # models = [("V1_EM", DFM_V1(D, K, 3)), ("V2_Autograd", DFM_V2(D, K, 3)), ("V3_Robust", DFM_V3(D, K, 3))]
        # models = [("NMF_ODE", DFM_ODE(D, K, 3)), ("LOU", OULatentModel(K, D)), ("V3_Robust", DFM_V3(D, K, 3))]
        models = [("NMF_ODE", DFM_ODE(D, K, 3))]
        for name, model in models:
            print(f"Training {name}...")
            if name == "LOU":
                # 1. Prepare JAX data
                lou_input, subj_lengths = bridge_to_jax(data, times)
                
                # 2. Run MCMC
                samples = model.fit(jax.random.PRNGKey(r), lou_input)
                
                # 3. Wrap with context for history calculations
                y_true_flattened = torch.cat(data, dim=0)
                wrapped_model = NumPyroModelWrapper(
                    samples, 
                    subj_lengths, 
                    gt_params=gt_params, 
                    y_true=y_true_flattened
                )
                trained_models["LOU"].append(wrapped_model)
            else:
                model.fit(data, covs, times, gt_params, epochs=epochs)
                trained_models[name].append(model)
    viz.plot_multi_model_metrics(trained_models)
    viz.plot_multi_model_trajectories(trained_models)
    viz.plot_loading_recovery(trained_models)


if __name__ == "__main__":
    run_benchmark()