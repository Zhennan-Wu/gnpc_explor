import os
import json
import numpy as np
import pandas as pd
import cmdstanpy
from scipy.linalg import expm
from scipy.special import expit
import os
import argparse
import arviz as az
import re

# os.environ["CC"] = "/usr/bin/gcc"
# os.environ["CXX"] = "/usr/bin/g++"

# cmdstanpy.install_cmdstan(verbose=True)

# ==========================================
# 1. Data Generation
# ==========================================
def generate_lou_simulation_data(N=600, scenario='S1', random_state=42):
    """
    Generates data for the Latent Ornstein-Uhlenbeck (LOU) simulation study.
    Scenarios S1 to S5 represent increasing complexity in the latent mean structure.
    """
    np.random.seed(random_state)
    
    # Measurement Model (IRT) Parameters
    Lambda = np.array([
        [1.2, 0.0], [4.0, 0.0], [4.1, 0.0], 
        [0.0, 3.1], [0.0, 5.2], [0.0, 3.0], [0.0, 1.7]
    ])
    
    beta = np.array([
        [0.3,  0.5], [0.1,  0.2], [-0.1, 0.2], 
        [-0.2, 0.4], [0.3, -0.3], [-0.1,-0.2], [-0.2,-0.1]
    ])
    
    theta = [
        np.array([2.3]), np.array([2.6]), np.array([2.9]), 
        np.array([-4.0, -1.0, 2.7]), np.array([-7.5, -2.5, 2.6]), 
        np.array([-5.5, -2.7, 2.5]), np.array([-4.3, -1.0, 1.4])
    ]
    
    sigma_b = np.array([3.7, 3.4, 4.8, 3.1, 6.1, 5.1, 1.7])
    
    # Latent OU Process Parameters
    rho = 0.6
    Omega = np.array([[1.0, rho], [rho, 1.0]])
    
    if scenario == 'S1':
        Gamma = np.array([[0.18, -0.07], [-0.10, 0.15]])
    elif scenario in ['S2', 'S3', 'S4', 'S5']:
        Gamma = np.array([[0.18, -0.07], [0.10, 0.15]])
    else:
        raise ValueError("Scenario must be between 'S1' and 'S5'")
        
    c_latent = np.array([0.5, -0.3])
    A_latent = np.array([[0.4, -0.2], [-0.3, 0.5]])
    B_latent = np.array([[0.1, 0.2], [-0.1, 0.3]])
        
    # Missing Data Parameters (MAR)
    K_miss = np.array([
        [-0.25, -0.27, -0.6, -1.8], [-0.24,  0.12, -0.4, -1.8],
        [-0.27, -0.11, -0.7, -1.5], [-0.22, -0.20, -0.7, -1.7],
        [ 0.25, -0.16, -0.7, -1.8], [ 0.20, -0.13, -0.8, -1.9],
        [ 0.25,  0.06, -1.1, -1.5]
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
        
        xi_star_history = np.zeros((n_i, 2)) 
        xi_i = np.zeros((n_i, 2))            
        Y_i = np.zeros((n_i, 7))
        Y_i_missing = np.zeros((n_i, 7))
        
        for j in range(n_i):
            x_ij = np.array([np.random.binomial(1, 0.5), np.random.normal(0, 1)])
            
            if j == 0:
                xi_star = np.random.multivariate_normal([0, 0], Omega)
            else:
                delta_t = d_ij[j]
                transition_mat = expm(-Gamma * delta_t)
                mean_xi = transition_mat @ xi_star_history[j-1]
                cov_xi = Omega - (transition_mat @ Omega @ transition_mat.T)
                xi_star = np.random.multivariate_normal(mean_xi, cov_xi)
                
            xi_star_history[j] = xi_star
            
            # Apply Mean Shifts based on Scenario
            if scenario in ['S1', 'S2']:
                trend_offset = np.zeros(2)
            elif scenario == 'S3':
                trend_offset = c_latent * t_ij[j]
            elif scenario == 'S4':
                trend_offset = (A_latent @ Z_i) * t_ij[j]
            elif scenario == 'S5':
                trend_offset = (A_latent @ Z_i) * t_ij[j] + (B_latent @ Z_i) + c_latent
                
            xi_i[j] = xi_star + trend_offset
            
            for k in range(7):
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
                for k in range(7):
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
                'xi1': xi_i[j, 0], 'xi2': xi_i[j, 1],
                'Y1': Y_i_missing[j, 0], 'Y2': Y_i_missing[j, 1], 'Y3': Y_i_missing[j, 2],
                'Y4': Y_i_missing[j, 3], 'Y5': Y_i_missing[j, 4], 'Y6': Y_i_missing[j, 5],
                'Y7': Y_i_missing[j, 6]
            })
            
    return dataset

# ==========================================
# 2. General Ground Truth & Data Prep
# ==========================================
def add_param_to_dict(d, name, val):
    """Programmatically unrolls arrays into Stan-compatible string keys."""
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

def create_ground_truth_dict(scenario='S1'):
    """Dynamically builds the ground truth dictionary based on the scenario."""
    truths = {}
    
    # 1. Base IRT Parameters (Shared across all)
    truths['theta1'] = 2.3
    truths['theta2'] = 2.6
    truths['theta3'] = 2.9
    add_param_to_dict(truths, 'theta4', np.array([-4.0, -1.0, 2.7]))
    add_param_to_dict(truths, 'theta5', np.array([-7.5, -2.5, 2.6]))
    add_param_to_dict(truths, 'theta6', np.array([-5.5, -2.7, 2.5]))
    add_param_to_dict(truths, 'theta7', np.array([-4.3, -1.0, 1.4]))
    
    add_param_to_dict(truths, 'lambda', np.array([1.2, 4.0, 4.1, 3.1, 5.2, 3.0, 1.7]))
    add_param_to_dict(truths, 'beta', np.array([
        [0.3, 0.5], [0.1, 0.2], [-0.1, 0.2], [-0.2, 0.4], 
        [0.3, -0.3], [-0.1, -0.2], [-0.2, -0.1]
    ]))
    add_param_to_dict(truths, 'sigma_bk', np.array([3.7, 3.4, 4.8, 3.1, 6.1, 5.1, 1.7]))
    truths['rho'] = 0.6
    
    # 2. Latent Drift (Gamma)
    if scenario == 'S1':
        add_param_to_dict(truths, 'Gamma', np.array([[0.18, -0.07], [-0.10, 0.15]]))
    else:
        add_param_to_dict(truths, 'Gamma', np.array([[0.18, -0.07], [0.10, 0.15]]))
        
    # 3. Dynamic Mean Parameters
    if scenario in ['S3', 'S5']:
        add_param_to_dict(truths, 'c_latent', np.array([0.5, -0.3]))
        
    if scenario in ['S4', 'S5']:
        add_param_to_dict(truths, 'A_latent', np.array([[0.4, -0.2], [-0.3, 0.5]]))
        
    if scenario == 'S5':
        add_param_to_dict(truths, 'B_latent', np.array([[0.1, 0.2], [-0.1, 0.3]]))
            
    return truths

def prepare_stan_data(dataset):
    df = pd.DataFrame(dataset)
    df['deltat'] = df.groupby('id')['time'].diff().fillna(0.0)
    repme = df.groupby('id').size().values
    cumu = np.cumsum(repme)
    
    Y_raw = df[['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7']].values
    missing_ID = np.isnan(Y_raw).astype(int)
    Y = np.nan_to_num(Y_raw, nan=-99).astype(int)
    
    for col_idx in range(3, 7):
        valid_mask = (missing_ID[:, col_idx] == 0)
        Y[valid_mask, col_idx] += 1
        
    stan_data = {
        'N': len(df), 'Nsub': df['id'].nunique(), 'K': 7, 'R': 2, 'p': 2, 'q': 2,
        'ID': df['id'].values.astype(int), 'cumu': cumu.astype(int), 'repme': repme.astype(int),
        'Y': Y, 'missing_ID': missing_ID, 'deltat': df['deltat'].values, 
        'time': df['time'].values,
        'X': df[['x1', 'x2']].values,
        'Z': df[['Z1', 'Z2']].fillna(0.0).values, 
        'ncate4': 4, 'ncate5': 4, 'ncate6': 4, 'ncate7': 4
    }
    return stan_data

# ==========================================
# 3. Execution Pipeline
# ==========================================
def evaluate_model_performance(stan_file_path, dataset, scenario='S1', iter_sampling=1000, iter_warmup=1000, chains=3):
    stan_data = prepare_stan_data(dataset)
    ground_truths = create_ground_truth_dict(scenario)
    
    print(f"Compiling and running {stan_file_path} against Scenario {scenario} data...")
    model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
    fit = model.sample(
        data=stan_data, iter_warmup=iter_warmup, iter_sampling=iter_sampling,
        chains=chains, parallel_chains=chains, adapt_delta=0.95, max_treedepth=12
    )
    
    print("Calculating performance metrics for overlapping parameters...")
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
    results = []
    
    for param_name, true_val in ground_truths.items():
        # THE SAFEGUARD: Only evaluates if both the ground truth AND the Stan model share the parameter
        if param_name in summary_df.index:
            est_mean = summary_df.loc[param_name, 'Mean']
            lower_ci = summary_df.loc[param_name, '2.5%']
            upper_ci = summary_df.loc[param_name, '97.5%']
            rhat = summary_df.loc[param_name, 'R_hat']
            # Safely grab ESS depending on the CmdStan version
            if 'ESS_bulk' in summary_df.columns:
                ess = summary_df.loc[param_name, 'ESS_bulk']
            elif 'N_Eff' in summary_df.columns:
                ess = summary_df.loc[param_name, 'N_Eff']
            else:
                ess = np.nan
            
            rbias = ((est_mean - true_val) / true_val) if true_val != 0 else np.nan
            mse = (est_mean - true_val) ** 2
            coverage = 1 if (lower_ci <= true_val <= upper_ci) else 0
            
            results.append({
                'Parameter': param_name, 'True_Value': true_val, 'Estimate': est_mean,
                '2.5%': lower_ci, '97.5%': upper_ci, 'R_hat': rhat, 'ESS': ess,
                'Rbias': rbias, 'MSE': mse, 'Coverage': coverage
            })
            
    metrics_df = pd.DataFrame(results)
    model_name = os.path.splitext(os.path.basename(stan_file_path))[0]
    
    output_filename_csv = f"results_{scenario}_{model_name}.csv"
    print(f"Saving detailed metrics to: {output_filename_csv}")
    metrics_df.to_csv(output_filename_csv, index=False)
    
    # Safely compute Aggregated Summary Metrics
    agg_summary = {
        'Average_MSE': float(metrics_df['MSE'].mean()) if not metrics_df.empty else None,
        'Average_Absolute_Rbias': float(metrics_df['Rbias'].abs().mean()) if not metrics_df.empty else None,
        'Global_Coverage_Prob': float(metrics_df['Coverage'].mean()) if not metrics_df.empty else None,
        'Max_Rhat': float(metrics_df['R_hat'].max()) if not metrics_df.empty else None,
        'Min_ESS': float(metrics_df['ESS'].min()) if not metrics_df.empty else None
    }
    
    def compute_matrix_corr(prefix):
        subset = metrics_df[metrics_df['Parameter'].str.startswith(prefix)]
        # Safeguard: prevents crash if matrix is entirely missing or has no variance
        if len(subset) > 1 and subset['True_Value'].std() > 0:
            return float(np.corrcoef(subset['True_Value'], subset['Estimate'])[0, 1])
        return None

    # Structural matrices
    agg_summary['Corr_Gamma'] = compute_matrix_corr('Gamma[')
    agg_summary['Corr_Beta'] = compute_matrix_corr('beta[')
    agg_summary['Corr_Lambda'] = compute_matrix_corr('lambda[')
    agg_summary['Corr_Theta'] = compute_matrix_corr('theta')
    agg_summary['Corr_Sigma_bk'] = compute_matrix_corr('sigma_bk[')
    
    # New shift matrices (safely handled if model lacks them)
    agg_summary['Corr_A_latent'] = compute_matrix_corr('A_latent[')
    agg_summary['Corr_B_latent'] = compute_matrix_corr('B_latent[')
    agg_summary['Corr_c_latent'] = compute_matrix_corr('c_latent[')
    
    if len(metrics_df) > 1:
        agg_summary['Corr_Global'] = float(np.corrcoef(metrics_df['True_Value'], metrics_df['Estimate'])[0, 1])
    else:
        agg_summary['Corr_Global'] = None
        
    output_filename_json = f"summary_{scenario}_{model_name}.json"
    print(f"Saving aggregated summary to: {output_filename_json}")
    with open(output_filename_json, 'w') as f:
        json.dump(agg_summary, f, indent=4)
    
    return metrics_df, agg_summary, fit

# ==========================================
# 4. Example Usage
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOU Simulation")
    parser.add_argument("--scenario", type=str, required=True, help="Data generation scenario")
    parser.add_argument("--model", type=str, required=True, help="Path to the Stan model file")
    parser.add_argument("--chains", type=int, default=3, help="Number of MCMC chains")
    
    # --- ADD THESE TWO LINES ---
    parser.add_argument("--warmup", type=int, default=2500, help="Warmup iterations")
    parser.add_argument("--sampling", type=int, default=2500, help="Sampling iterations")
    # ---------------------------
    
    args = parser.parse_args()
    
    scenario_to_run = args.scenario
    stan_file = args.model
    num_chains = args.chains
    
    print(f"Starting run: Scenario = {scenario_to_run}, Model = {stan_file}, Chains = {num_chains}")
    
    dataset = generate_lou_simulation_data(N=600, scenario=scenario_to_run)
    
    metrics_df, agg_summary, fit_object = evaluate_model_performance(
        stan_file_path=stan_file, 
        dataset=dataset, 
        scenario=scenario_to_run,
        iter_warmup=args.warmup,       # <-- Pass the argument here
        iter_sampling=args.sampling,   # <-- Pass the argument here
        chains=num_chains
    )
    
    print(f"\nRun Complete for {stan_file} on {scenario_to_run} Data.")