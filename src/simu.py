import torch
import jax
import jax.numpy as jnp
import numpy as np
import json
from dfa_ou import OUDynamicFactorModel as DFM_V1
from dfa_ou_autograd import OUDynamicFactorModel as DFM_V2
from dfa_ou_damp import OUDynamicFactorModel as DFM_V3
from nmf_ode_l1reg import NMF_LinearODE_Model as NMF_ODE_V1
from nmf_ode_cov_l1reg import NMF_LinearODE_Model as NMF_ODE_V2

from lou_test import OULatentModel
from comp_utils import NumPyroModelWrapper, bridge_to_jax
from visual import ModelVisualizer
import os


def generate_parameters(D, K, M, L=None, scenario='default'):
    """
    Generates a ground truth parameter dictionary based on the model constraints.
    
    Scenarios:
        - 'default': Balanced dynamics.
        - 'high_persistence': Small Theta values (slow mean reversion).
        - 'high_noise': Large Psi values (noisy observations).
    """
    if L is None:
        L = K  # Default number of uncertainty sources to match latent dimension
    
    # 1. Loading Matrix Lambda (D x K) - using Xavier-style initialization
    Lambda = np.random.randn(D, K) * np.sqrt(1 / K)
    
    # 2. OU Mean Reversion Speeds Theta_diag (K,) - Must be positive
    if scenario == 'high_persistence':
        Theta_diag = np.random.uniform(0.1, 0.5, K) # Slow decay
    else:
        Theta_diag = np.random.uniform(1.0, 5.0, K) # Faster decay
        
    # 3. Diffusion Scale Gamma (K x L) and Q = Gamma @ Gamma.T
    Gamma = np.random.randn(K, L) * 0.5
    
    # 4. Covariate Effects Phi (K x M) and Population Shift alpha (K,)
    Phi = np.random.randn(K, M) * 0.5
    alpha = np.random.randn(K) * 0.1
    
    # 5. Observation Noise Psi (D,)
    psi_scale = 2.0 if scenario == 'high_noise' else 0.5
    Psi = np.ones(D) * psi_scale
    
    # 6. Initial State p0: f(0) ~ N(f0_mean, f0_cov)
    f0_mean = np.zeros(K)
    f0_cov = np.eye(K) * 0.5
    
    return {
        'Lambda': Lambda,
        'Theta_diag': Theta_diag,
        'Gamma': Gamma,
        'Phi': Phi,
        'alpha': alpha,
        'Psi': Psi,
        'f0_mean': f0_mean,
        'f0_cov': f0_cov
    }


def generate_dynamic_factor_data(D, K, M, T_range, N_subjects, time_gap_range, gt_params):
    """
    Generates data from a Dynamic Factor Model with an OU latent process.
    T_range: Tuple (min_visits, max_visits) for irregular sequence lengths.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T_min, T_max = T_range

    # --- 1. Consistency and Dimensionality Checks ---
    required_keys = ['Lambda', 'Psi', 'Theta_diag', 'Gamma', 'Phi', 'alpha', 'f0_mean', 'f0_cov']
    for key in required_keys:
        if key not in gt_params:
            raise KeyError(f"Missing required ground truth parameter: {key}")

    Lambda = gt_params['Lambda']
    Theta_diag = gt_params['Theta_diag']
    Gamma = gt_params['Gamma']
    Phi = gt_params['Phi']
    alpha = gt_params['alpha']
    Q = Gamma @ Gamma.T  
    
    # Pre-process Psi
    if gt_params['Psi'].ndim == 1:
        Psi_mat = np.diag(gt_params['Psi'])
    else:
        Psi_mat = gt_params['Psi']

    # --- 2. Data Generation ---
    # We use lists to store subjects since T varies per subject
    X_list, F_list, S_list, Times_list = [], [], [], []

    for i in range(N_subjects):
        # Randomly sample number of visits for this subject
        T_i = np.random.randint(T_min, T_max + 1)
        
        # Generate Covariates s_i
        s_i = np.random.normal(0, 1, M)
        S_list.append(torch.from_numpy(s_i).float().to(device))
        
        # Generate Irregular Time Visits
        # gaps = np.random.uniform(time_gap_range[0], time_gap_range[1], T_i - 1)
        gaps = np.random.randint(time_gap_range[0], time_gap_range[1] + 1, size=T_i - 1)
        t_steps = np.concatenate(([0], np.cumsum(gaps)))
        Times_list.append(torch.from_numpy(t_steps).float().to(device))
        
        # Initialize containers for this subject
        f_subject = np.zeros((T_i, K))
        x_subject = np.zeros((T_i, D))
        
        # Initial State
        f_current = np.random.multivariate_normal(gt_params['f0_mean'], gt_params['f0_cov'])
        f_subject[0] = f_current
        
        # Initial Observation
        epsilon_0 = np.random.multivariate_normal(np.zeros(D), Psi_mat)
        x_subject[0] = Lambda @ f_current + epsilon_0

        for j in range(1, T_i):
            dt = t_steps[j] - t_steps[j-1]
            
            # OU Process Transition
            A_ij_diag = np.exp(-Theta_diag * dt)
            mu_i = alpha + Phi @ s_i
            b_ij = (1.0 - A_ij_diag) * mu_i
            
            # Transition Covariance (Exact OU solution)
            Sigma_ij = np.zeros((K, K))
            for k1 in range(K):
                for k2 in range(K):
                    denom = Theta_diag[k1] + Theta_diag[k2]
                    if denom < 1e-8:  # Avoid division by zero
                        Sigma_ij[k1, k2] = 0.0
                    else:
                        Sigma_ij[k1, k2] = Q[k1, k2] * (1 - np.exp(-denom * dt)) / denom
            
            # Sample next latent state
            mean_f = A_ij_diag * f_current + b_ij
            f_current = np.random.multivariate_normal(mean_f, Sigma_ij)
            f_subject[j] = f_current
            
            # Generate Observation
            epsilon = np.random.multivariate_normal(np.zeros(D), Psi_mat)
            x_subject[j] = Lambda @ f_subject[j] + epsilon

        X_list.append(torch.from_numpy(x_subject).float().to(device))
        F_list.append(torch.from_numpy(f_subject).float().to(device))

    # Prepare ground truth params for return
    params_to_eval = {
        'Lambda': torch.from_numpy(Lambda).float().to(device), 
        'rho': torch.from_numpy(Theta_diag).float().to(device), 
        'Gamma': torch.from_numpy(Gamma).float().to(device), 
        'alpha': torch.from_numpy(alpha).float().to(device), 
        'sigma_obs': torch.tensor(Psi_mat[0][0]).float().to(device), 
        'A': torch.from_numpy(np.diag(Theta_diag)).float().to(device), 
        'Phi': torch.from_numpy(Phi).float().to(device)
    }

    return {
        "observations": X_list,
        "timestamps": Times_list,
        "covariates": S_list,
        "gt_params": params_to_eval,
        "latent_factors": F_list
    }


def generate_simulation_data_automated(D = 20 , K = 3 , M = 3 , T_range = (2, 10) , N_subjects = 50 , time_gaps = (0, 5.0)):
    """
    Example execution: Define dimensions, generate params, and create data.
    """
    
    print(f"--- Generating Data for D={D}, K={K}, M={M} ---")
    
    try:
        # Step 1: Generate Parameters
        gt_params = generate_parameters(D, K, M, scenario='default')
        
        # Step 2: Generate Data using the function from previous response
        data_bundle = generate_dynamic_factor_data(
            D, K, M, T_range, N_subjects, time_gaps, gt_params
        )
        
        # Validation Output
        obs = data_bundle['observations']
        latents = data_bundle['latent_factors']
        print(f"Successfully generated data!")

        return data_bundle['observations'], data_bundle['timestamps'], data_bundle['covariates'], data_bundle['gt_params'], data_bundle['latent_factors']

    except (ValueError, KeyError) as e:
        print(f"Consistency Check Failed: {e}")


def generate_simulation_data_manually():
    """
    Example execution: Define dimensions, generate params, and create data.
    """
    D = 5
    K = 2
    M = 3
    T_range = (2, 10)
    N_subjects = 10
    time_gaps = (1, 5)
    Lambda = np.array([[0.5, 0.1], [0.1, 0.5], [0.3, 0.3], [0.1, 0.2], [0.1, 0.1]])
    Theta_diag = np.array([2, 4])
    Gamma = np.array([[0., 0.], [0., 0.]])
    Phi = np.array([[0.1, 0.8, -0.2], [-0.1, -0.5, 0.6]])
    alpha = np.array([0.1, 0.2])
    Psi = np.eye(D) * 0.01
    f0_mean = np.zeros(K)
    f0_cov = np.eye(K) * 0.1

    print(f"--- Generating Data for D={D}, K={K}, M={M} ---")
    
    try:
        # Step 1: Generate Parameters
        gt_params = {
            'Lambda': Lambda,
            'Theta_diag': Theta_diag,
            'Gamma': Gamma,
            'Phi': Phi,
            'alpha': alpha,
            'Psi': Psi,
            'f0_mean': f0_mean,
            'f0_cov': f0_cov            
        }
        
        # Step 2: Generate Data using the function from previous response
        data_bundle = generate_dynamic_factor_data(
            D, K, M, T_range, N_subjects, time_gaps, gt_params
        )
        
        # Validation Output
        print(f"Successfully generated data!")

        return data_bundle['observations'], data_bundle['timestamps'], data_bundle['covariates'], data_bundle['gt_params'], data_bundle['latent_factors']

    except (ValueError, KeyError) as e:
        print(f"Consistency Check Failed: {e}")


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
    Phi = torch.zeros(K, 3)
    Phi[:, 2] = 0.3
    gamma = torch.eye(K, device=device) * (0.05 ** 2)

    gt_params = {'Lambda': Lambda, 'rho': rho, 'Gamma': gamma, 'alpha': drift_base, 'sigma_obs': torch.tensor(noise_std, device=device), 'A': -torch.diag(rho), 'b': drift_base * rho, 'Phi': Phi}

    return all_data, all_times, all_covs, gt_params, all_factors


def run_benchmark(D=5, K=2, T_range=(2, 10), subjects=10, n_runs=3, epochs=500, save_dir="../viz_results"):
    # data, times, covs, gt_params, factors = generate_data(D, K, T, subjects)
    T = 5
    # data, times, covs, gt_params, factors = generate_disease_proteomics_data(D, K, T, subjects)
    # data, times, covs, gt_params, factors = generate_simulation_data_automated(D=D, K=K, T_range=T_range, N_subjects=subjects)
    data, times, covs, gt_params, factors = generate_simulation_data_manually()
    simu_data = {"D": D, "K": K, "T_range": T_range, "N_subjects": subjects, "observations": data, "timestamps": times, "covariates": covs, "gt_params": gt_params, "latent_factors": factors}
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/simu_data.json", "w") as f:
        json.dump(simu_data, f, default=lambda x: x.cpu().numpy().tolist() if isinstance(x, torch.Tensor) else x, indent=4, sort_keys=True)

    gt_data = {'traj': factors, 'times': times, 'data': data, 'covs': covs}
    viz = ModelVisualizer(gt_params, gt_data, save_dir=save_dir)
    # trained_models = {name: [] for name in ["V1_EM", "V2_Autograd", "V3_Robust"]}
    # trained_models = {name: [] for name in ["V3_Robust", "V2_Autograd"]}
    trained_models = {name: [] for name in ["NMF_ODE_Sub", "NMF_ODE_Cov"]}
    for r in range(n_runs):
        print(f"--- Run {r+1}/{n_runs} ---")
        # models = [("V1_EM", DFM_V1(D, K, 3)), ("V2_Autograd", DFM_V2(D, K, 3)), ("V3_Robust", DFM_V3(D, K, 3)), ("LOU", OULatentModel(K, D))]
        # models = [("V3_Robust", DFM_V3(D, K, 3)), ("V2_Autograd", DFM_V2(D, K, 3))]
        models = [("NMF_ODE_Sub", NMF_ODE_V1(D, K, 3)), ("NMF_ODE_Cov", NMF_ODE_V2(D, K, 3))]
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