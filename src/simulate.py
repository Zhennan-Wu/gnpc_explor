import torch
import numpy as np

def generate_multi_setting_data(
    N=15, 
    D=100, 
    K=5, 
    M=2, 
    snr=1.0, 
    obs_noise_exists=True, 
    dynamic_noise_exists=True,
    irregular_intervals=True,
    sparsity=0.7
):
    """
    Generates synthetic longitudinal data to test model robustness.
    
    Parameters:
    - N: Number of subjects[cite: 17, 246].
    - D: Observation dimension (proteomics features)[cite: 17, 247].
    - K: Number of latent factors[cite: 27, 248].
    - M: Number of covariates[cite: 111, 117].
    - snr: Signal-to-Noise Ratio (scales the variance of Lambda f relative to Psi).
    - obs_noise_exists: If False, Psi (observation noise) is near zero[cite: 29, 286].
    - dynamic_noise_exists: If False, Gamma (innovation noise) is near zero[cite: 105, 114].
    """
    device = 'cpu'
    
    # 1. Generate Sparse Loading Matrix Lambda [cite: 31, 288-291]
    Lambda_true = torch.randn(D, K)
    mask = torch.rand(D, K) > sparsity
    Lambda_true = Lambda_true * mask
    # Identifiability: Leading KxK block lower triangular [cite: 66]
    for i in range(K):
        for j in range(i + 1, K):
            Lambda_true[j, i] = 0.0
    # Normalize columns [cite: 293]
    Lambda_true = Lambda_true / (torch.norm(Lambda_true, dim=0) + 1e-9)
    
    # 2. OU Parameters [cite: 110, 266-267]
    rho_true = 0.05 + 0.03 * torch.randn(K).abs() # Decay rates
    # Scale dynamic noise [cite: 114, 273-277]
    sigma_state_base = 0.1 if dynamic_noise_exists else 1e-6
    sigma_state_true = torch.ones(K) * sigma_state_base
    
    # 3. Observation Noise (Psi) [cite: 29, 298-299]
    obs_var_base = (1.0 / snr) if obs_noise_exists else 1e-6
    psi_diag = obs_var_base * torch.exp(torch.randn(D) * 0.5) # Heteroskedastic
    
    data, times, covs, latent_paths = [], [], [], []
    
    for i in range(N):
        Ji = np.random.randint(5, 12) # Number of measurements [cite: 17]
        
        # Sampling intervals [cite: 11, 251-252]
        if irregular_intervals:
            dt = 0.3 + (5.0 - 0.3) * torch.rand(Ji)
        else:
            dt = torch.ones(Ji) * 1.5
        t = torch.cumsum(dt, dim=0)
        
        s_i = torch.randn(M) # Covariates [cite: 117]
        
        # Simulate Latent Process f_i(t) [cite: 102, 258-261]
        f = torch.zeros(Ji, K)
        f_curr = torch.randn(K) # Initial distribution [cite: 104, 109]
        
        for j in range(Ji):
            dt_step = dt[j] if j > 0 else 0.1
            A = torch.exp(-rho_true * dt_step) # [cite: 143, 158]
            Q = (sigma_state_true / (2 * rho_true)) * (1 - torch.exp(-2 * rho_true * dt_step)) # [cite: 161]
            # Innovation step [cite: 105, 147-148]
            f_curr = A * f_curr + torch.randn(K) * torch.sqrt(Q)
            f[j] = f_curr
            
        # 4. Generate Observations x_i(t) = Lambda f_i(t) + epsilon [cite: 29, 286]
        signal = f @ Lambda_true.T
        noise = torch.randn(Ji, D) * torch.sqrt(psi_diag)
        x = signal + noise
        
        data.append(x)
        times.append(t)
        covs.append(s_i)
        latent_paths.append(f)
        
    return {
        'data': data,
        'times': times,
        'covs': covs,
        'L_true': Lambda_true,
        'f_true': latent_paths,
        'psi_true': psi_diag
    }


def generate_nmf_ode_data(N=15, D=20, K=3):
    # 1. Non-negative Loadings (NMF style)
    Lambda_true = torch.abs(torch.randn(D, K)) 
    Lambda_true[torch.rand(D, K) < 0.6] = 0.0 # Sparse [cite: 289]
    
    data, times, covs = [], [], []
    for i in range(N):
        Ji = 8
        t = torch.linspace(0.1, 10, Ji) # Regular/Irregular [cite: 251]
        s_i = torch.randn(2)
        
        # 2. ODE-like Latent path (e.g., Logistic growth or Exponential decay)
        # Instead of SDE noise, we use a deterministic function
        f = torch.zeros(Ji, K)
        for k in range(K):
            # Deterministic decay/growth to simulate ODE
            f[:, k] = torch.exp(-0.2 * t) * (k + 1) 
            
        # 3. Non-negative observations
        # x(t) = Lambda * f(t) + small noise 
        x = f @ Lambda_true.T + torch.abs(torch.randn(Ji, D) * 0.05)
        
        data.append(x); times.append(t); covs.append(s_i)
        
    return data, times, covs, Lambda_true