import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from dfa_ou import OUDynamicFactorModel as DFM_V1
from dfa_ou_autograd import OUDynamicFactorModel as DFM_V2
from dfa_ou3 import BestOfBothOUDFM as DFM_V3

def generate_data(D, K, T, N_subjects, noise_std=0.5, sparsity=0.5):
    """
    Generates synthetic OU-process data.
    Scenarios:
    - Low D vs High D (Scalability)
    - Sparse vs Dense Loadings (Horseshoe performance)
    - High Noise vs Low Noise (Robustness)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Generate Sparse Loadings (Lambda)
    Lambda = torch.randn(D, K, device=device)
    mask = torch.rand(D, K, device=device) > sparsity
    Lambda = Lambda * mask.float()
    
    # 2. Latent OU Process: f_t = exp(-rho*dt)*f_{t-1} + noise
    rho = torch.rand(K, device=device) * 0.5 + 0.1
    all_data, all_times, all_covs = [], [], []
    
    for _ in range(N_subjects):
        times = torch.cumsum(torch.rand(T, device=device) * 0.5 + 0.1, dim=0)
        factors = torch.zeros(T, K, device=device)
        f_curr = torch.randn(K, device=device)
        
        for t in range(T):
            if t > 0:
                dt = times[t] - times[t-1]
                f_curr = torch.exp(-rho * dt) * f_curr + torch.randn(K, device=device) * 0.1
            factors[t] = f_curr
            
        # Observation: Y = Lambda @ f + epsilon
        obs = factors @ Lambda.T + torch.randn(T, D, device=device) * noise_std
        
        all_data.append(obs)
        all_times.append(times)
        all_covs.append(torch.randn(3, device=device)) # Dummy covariates

    return all_data, all_times, all_covs, Lambda

def run_benchmark(scenario_name, D, K, T, subjects, epochs=50):
    print(f"\n--- Scenario: {scenario_name} (D={D}, K={K}, T={T}) ---")
    data, times, covs, L_true = generate_data(D, K, T, subjects)
    models = [
        ("V1: Standard EM", DFM_V1(D, K, 3)),
        ("V2: Pure Autograd", DFM_V2(D, K, 3)),
        ("V3: Damped/Robust", DFM_V3(D, K, 3))
    ]
    
    results = {}
    for name, model in models:
        start_time = time.time()
        try:
            # Standardize training epochs for comparison
            print(f"Start training {name}...")
            model.fit(data, covs, times, L_true, epochs=epochs)
            
            elapsed = time.time() - start_time
            corr, mse = model.evaluate(L_true, data, times, covs)
            results[name] = {"Time": elapsed, "Corr": corr, "MSE": mse}
            print(f"{name} -> Time: {elapsed:.2f}s | Corr: {corr:.4f} | MSE: {mse:.4f}")
        except Exception as e:
            print(f"{name} Failed: {e}")
            
    return results

if __name__ == "__main__":
    # Test 1: Scalability (High Dimension)
    run_benchmark("High Dimension", D=8000, K=20, T=5, subjects=100)
    
    # Test 2: Low Signal (High Noise)
    run_benchmark("Noisy Data", D=8000, K=10, T=5, subjects=100)