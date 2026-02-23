import torch
import numpy as np
import time
from dfa_ou import OUDynamicFactorModel as DFM_V1
from dfa_ou_autograd import OUDynamicFactorModel as DFM_V2
from dfa_ou3 import BestOfBothOUDFM as DFM_V3
from nmf_ode2 import NMF_LinearODE_Model

def generate_data(D, K, T, N_subjects, process_noise=0.1, obs_noise=0.5, non_negative=False):
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

def run_benchmark(scenario_name, params, epochs=500):
    print(f"\n--- Scenario: {scenario_name} ---")
    data, times, covs, L_true = generate_data(**params)
    
    models = [
        ("NMF-ODE", NMF_LinearODE_Model(params['D'], params['K'], 3)), 
        ("V1: Standard EM", DFM_V1(params['D'], params['K'], 3)),
        ("V2: Pure Autograd", DFM_V2(params['D'], params['K'], 3)),
        ("V3: Damped/Robust", DFM_V3(params['D'], params['K'], 3))
    ]
    
    for name, model in models:
        try:
            start = time.time()
            print(f"Start training {name}...")
            model.fit(data, covs, times, L_true, epochs=epochs)
            
            elapsed = time.time() - start
            corr, mse = model.evaluate(L_true, data, times, covs)
            print(f"{name} -> Time: {elapsed:.2f}s | Corr: {corr:.4f} | MSE: {mse:.4f}")
        except Exception as e:
            print(f"{name} Failed: {e}")
        break

if __name__ == "__main__":
    # Case 1: Stochastic & Real-valued (DFA+OU is correctly specified)
    run_benchmark("Stochastic/Real", {
        'D': 100, 'K': 5, 'T': 50, 'N_subjects': 5,
        'process_noise': 0.1, 'obs_noise': 0.5, 'non_negative': False
    })

    # Case 2: Deterministic & Non-negative (NMF+ODE is correctly specified)
    run_benchmark("Deterministic/NMF", {
        'D': 100, 'K': 5, 'T': 50, 'N_subjects': 5,
        'process_noise': 0.0, 'obs_noise': 0.05, 'non_negative': True
    })