import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os
import argparse
import glob

# --- BIG RED 200 CPU OPTIMIZATION ---
# Restrict PyTorch to the number of CPUs allocated by SLURM to prevent node thrashing
try:
    num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    torch.set_num_threads(num_threads)
    if num_threads > 1:
        print(f"Restricted PyTorch to {num_threads} threads.")
except Exception as e:
    print("Could not set thread count automatically.")
# ------------------------------------

# ---------------------------------------------------------
# 1. GPU-Accelerated Universal DFOULS Model (Identity Noise + Multi-start)
# ---------------------------------------------------------
class Universal_DFOULS(nn.Module):
    def __init__(self, obs_dim, latent_dim, covar_dim, delta=1e-4, theta_mode="diagonal"):
        super().__init__()
        self.D = obs_dim
        self.K = latent_dim
        self.C_dim = covar_dim
        self.delta = delta
        self.theta_mode = theta_mode
        
        if self.theta_mode == "dense":
            self.L = nn.Parameter(torch.eye(self.K) + 0.1 * torch.randn(self.K, self.K))
            self.K_unconstrained = nn.Parameter(torch.randn(self.K, self.K) * 0.1)
        elif self.theta_mode == "diagonal":
            self.log_rho = nn.Parameter(torch.randn(self.K) * 0.1 - 2.0)
            
        self.B = nn.Parameter(torch.randn(self.K, self.C_dim) * 0.1)
        self.C_int = nn.Parameter(torch.randn(self.K, self.C_dim) * 0.1)
        self.d_bias = nn.Parameter(torch.randn(self.K) * 0.1)
        
        self.Z = nn.Parameter(torch.randn(self.D, self.K) - 0.5) 
        self.register_buffer('tril_mask', torch.tril(torch.ones(self.D, self.K)))

    def get_theta(self):
        if self.theta_mode == "dense":
            L_tril = torch.tril(self.L)
            S = L_tril @ L_tril.T + self.delta * torch.eye(self.K, device=self.Z.device)
            return S + (self.K_unconstrained - self.K_unconstrained.T)
        return torch.diag(torch.exp(self.log_rho))

    def get_subject_matrices(self, Theta, u, times):
        dt = times[1:] - times[:-1]
        Theta_batch = Theta.unsqueeze(0).expand(times.shape[0]-1, self.K, self.K)
        A_trans = torch.linalg.matrix_exp(-Theta_batch * dt.view(-1, 1, 1))
        
        u_t, t_val = u[1:], times[1:].unsqueeze(1)
        mu = u_t @ self.B.T + (u_t * t_val) @ self.C_int.T + self.d_bias
        
        I_batch = torch.eye(self.K, device=self.Z.device).unsqueeze(0).expand(times.shape[0]-1, self.K, self.K)
        b_shift = torch.bmm(I_batch - A_trans, mu.unsqueeze(-1)).squeeze(-1)
        
        Lambda = self.tril_mask * torch.exp(self.Z)
        return A_trans, b_shift, dt, Lambda

    def kalman_smoother(self, x_obs, A_trans, b_shift, dt, Lambda):
        T = x_obs.shape[0]
        device = x_obs.device
        
        f_pred, P_pred = torch.zeros(T, self.K, device=device), torch.zeros(T, self.K, self.K, device=device)
        f_filt, P_filt = torch.zeros(T, self.K, device=device), torch.zeros(T, self.K, self.K, device=device)
        f_filt[0], P_filt[0] = torch.zeros(self.K, device=device), torch.eye(self.K, device=device)
        
        I_k = torch.eye(self.K, device=device)
        
        # [OPTIMIZATION]: Since R is Identity, R^-1 is Identity.
        # M = Lambda^T * R^-1 * Lambda simplifies to Lambda^T @ Lambda (Shape: [K, K])
        M = Lambda.T @ Lambda 
        
        for j in range(1, T):
            idx = j - 1
            f_pred[j] = A_trans[idx] @ f_filt[j-1] + b_shift[idx]
            P_pred[j] = A_trans[idx] @ P_filt[j-1] @ A_trans[idx].T + (dt[idx] * I_k)
            
            if torch.isnan(x_obs[j]).all():
                f_filt[j], P_filt[j] = f_pred[j], P_pred[j]
            else:
                x_pred = Lambda @ f_pred[j]
                
                # [OPTIMIZATION]: Woodbury Identity
                # We only invert KxK matrices (20x20) instead of DxD (10,000x10,000)
                P_pred_inv = torch.linalg.inv(P_pred[j])
                P_filt[j] = torch.linalg.inv(P_pred_inv + M)
                
                # K_gain = P_filt @ Lambda^T @ R^-1 simplifies to P_filt @ Lambda^T
                K_gain = P_filt[j] @ Lambda.T  # Shape: [K, D]
                
                f_filt[j] = f_pred[j] + K_gain @ (x_obs[j] - x_pred)
            
        f_smooth, P_smooth, P_cross = torch.zeros_like(f_filt), torch.zeros_like(P_filt), torch.zeros_like(P_filt)
        f_smooth[-1], P_smooth[-1] = f_filt[-1], P_filt[-1]
        
        for j in range(T-2, -1, -1):
            J_t = P_filt[j] @ A_trans[j].T @ torch.linalg.inv(P_pred[j+1])
            f_smooth[j] = f_filt[j] + J_t @ (f_smooth[j+1] - f_pred[j+1])
            P_smooth[j] = P_filt[j] + J_t @ (P_smooth[j+1] - P_pred[j+1]) @ J_t.T
            P_cross[j+1] = J_t @ P_smooth[j+1]
            
        return f_smooth, P_smooth, P_cross

    def expected_complete_log_posterior_vectorized(self, subjects_data, smoothed_stats, Theta, Lambda):
        ll_obs, ll_lat = 0.0, 0.0
        LTL = Lambda.T @ Lambda 
        
        for i, subj in enumerate(subjects_data):
            x_obs, u, times = subj['x'], subj['u'], subj['t']
            f_s, P_s, P_c = smoothed_stats[i]
            A_trans, b_shift, dt, _ = self.get_subject_matrices(Theta, u, times)
            
            valid_mask = ~torch.isnan(x_obs).any(dim=1)
            if valid_mask.any():
                x_v, f_v, P_v = x_obs[valid_mask], f_s[valid_mask], P_s[valid_mask]
                trace_E = torch.sum(P_v * LTL, dim=(1,2)) + torch.sum(f_v * (f_v @ LTL), dim=1)
                term_obs = torch.sum((x_v**2), dim=1) - 2 * torch.sum(x_v * (f_v @ Lambda.T), dim=1) + trace_E
                ll_obs -= torch.sum(0.5 * term_obs)
                
            f_j, f_jm1 = f_s[1:], f_s[:-1]
            P_j, P_jm1, P_cj = P_s[1:], P_s[:-1], P_c[1:]
            
            trace_Ej = torch.diagonal(P_j, dim1=-2, dim2=-1).sum(-1) + torch.sum(f_j**2, dim=1)
            AtA = torch.bmm(A_trans.transpose(1, 2), A_trans)
            trace_AEjm1A = torch.sum(P_jm1 * AtA, dim=(1,2)) + torch.sum(f_jm1 * torch.bmm(AtA, f_jm1.unsqueeze(-1)).squeeze(-1), dim=1)
            bb = torch.sum(b_shift**2, dim=1)
            
            trace_AEcross = torch.sum(P_cj * A_trans, dim=(1,2)) + torch.sum(f_j * torch.bmm(A_trans, f_jm1.unsqueeze(-1)).squeeze(-1), dim=1)
            b_f = torch.sum(b_shift * f_j, dim=1)
            A_fm1 = torch.bmm(A_trans, f_jm1.unsqueeze(-1)).squeeze(-1)
            b_A_fm1 = torch.sum(b_shift * A_fm1, dim=1)
            
            expected_residual = trace_Ej + trace_AEjm1A + bb - 2*trace_AEcross - 2*b_f + 2*b_A_fm1
            ll_lat += torch.sum(-0.5 * self.K * torch.log(dt) - 0.5 * (1/dt) * expected_residual)
            
        active_Z = self.Z[self.tril_mask == 1]
        log_prior_Z = -0.5 * torch.sum(active_Z ** 2)
        
        log_prior_dyn = -0.5 * torch.sum(self.log_rho ** 2) if self.theta_mode == "diagonal" else \
                        -0.5 * (torch.sum(torch.tril(self.L) ** 2) + torch.sum(self.K_unconstrained ** 2))
        log_prior_lin = -0.5 * (torch.sum(self.B**2) + torch.sum(self.C_int**2) + torch.sum(self.d_bias**2))
        
        return ll_obs + ll_lat + log_prior_Z + log_prior_dyn + log_prior_lin 

    def pca_warm_start(self, subjects_data):
        with torch.no_grad():
            x_all = torch.cat([s['x'] for s in subjects_data], dim=0)
            x_valid = x_all[~torch.isnan(x_all).any(dim=1)] 
            U, S_vals, Vh = torch.linalg.svd(x_valid - x_valid.mean(dim=0), full_matrices=False)
            
            Lambda_pca = Vh[:self.K, :].T * torch.sqrt(S_vals[:self.K] / x_valid.shape[0])
            q, r = torch.linalg.qr(Lambda_pca.T)
            Lambda_tril = r.T * torch.sign(torch.diag(r.T)).unsqueeze(0)
            
            mask = self.tril_mask == 1
            self.Z.data[mask] = torch.log(torch.abs(Lambda_tril[mask]) + 1e-4)
            self.B.data.fill_(0.0); self.C_int.data.fill_(0.0); self.d_bias.data.fill_(0.0)

    def fit_em_multistart(self, subjects_data, num_em_epochs=40, m_step_iters=20, lr=0.005, n_starts=3, burn_in_epochs=10, verbose=False):
        best_loss = float('inf')
        best_state_dict = None
        
        if verbose:
            print(f"\n  --- Starting EM Optimization ({n_starts} multi-starts) ---")
        
        for start in range(n_starts):
            with torch.no_grad():
                if self.theta_mode == "dense":
                    nn.init.normal_(self.L, mean=0.0, std=0.1)
                    self.L.data += torch.eye(self.K, device=self.L.device)
                    nn.init.normal_(self.K_unconstrained, mean=0.0, std=0.1)
                else:
                    nn.init.normal_(self.log_rho, mean=-2.0, std=0.1)
                    
                nn.init.normal_(self.B, mean=0.0, std=0.1)
                nn.init.normal_(self.C_int, mean=0.0, std=0.1)
                nn.init.normal_(self.d_bias, mean=0.0, std=0.1)
                
                # Note: If you paste this into the 'het_multi' version, uncomment the line below:
                # nn.init.normal_(self.log_psi, mean=0.0, std=0.1)
            
            self.pca_warm_start(subjects_data)
            start_loss = 0.0
            
            # --- BURN-IN PHASE ---
            for epoch in range(burn_in_epochs):
                Theta, Lambda = self.get_theta(), self.tril_mask * torch.exp(self.Z)
                smoothed_stats = []
                with torch.no_grad():
                    for subj in subjects_data:
                        A_trans, b_shift, dt, _ = self.get_subject_matrices(Theta, subj['u'], subj['t'])
                        smoothed_stats.append(self.kalman_smoother(subj['x'], A_trans, b_shift, dt, Lambda))
                
                # FIX 1: Reset Adam optimizer every epoch to clear stale momentum
                optimizer = optim.Adam(self.parameters(), lr=lr)
                
                epoch_loss = 0.0
                for m in range(m_step_iters):
                    optimizer.zero_grad()
                    Theta_m, Lambda_m = self.get_theta(), self.tril_mask * torch.exp(self.Z)
                    loss = -self.expected_complete_log_posterior_vectorized(subjects_data, smoothed_stats, Theta_m, Lambda_m)
                    loss.backward()
                    
                    # FIX 2: Clip gradients to prevent exponential explosion
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                start_loss = epoch_loss / m_step_iters
                
                if verbose and (epoch + 1) % max(1, burn_in_epochs // 2) == 0:
                    print(f"    [Start {start+1}/{n_starts}] Burn-in Epoch {epoch+1}/{burn_in_epochs} | Loss: {start_loss:.4f}")
                
            if start_loss < best_loss:
                best_loss = start_loss
                best_state_dict = {k: v.clone() for k, v in self.state_dict().items()}
                if verbose:
                    print(f"    --> New best start found! Loss: {best_loss:.4f}")
                
        self.load_state_dict(best_state_dict)
        
        if verbose:
            print(f"  --- Proceeding with Best Start (Main EM Phase) ---")
            
        final_loss = best_loss
        loss_history = [final_loss]
        main_epochs = num_em_epochs - burn_in_epochs
        
        # --- MAIN EM PHASE ---
        for epoch in range(main_epochs):
            Theta, Lambda = self.get_theta(), self.tril_mask * torch.exp(self.Z)
            smoothed_stats = []
            with torch.no_grad():
                for subj in subjects_data:
                    A_trans, b_shift, dt, _ = self.get_subject_matrices(Theta, subj['u'], subj['t'])
                    smoothed_stats.append(self.kalman_smoother(subj['x'], A_trans, b_shift, dt, Lambda))
            
            # FIX 1: Reset Adam optimizer every epoch
            optimizer = optim.Adam(self.parameters(), lr=lr)
            
            epoch_loss = 0.0
            for m in range(m_step_iters):
                optimizer.zero_grad()
                Theta_m, Lambda_m = self.get_theta(), self.tril_mask * torch.exp(self.Z)
                loss = -self.expected_complete_log_posterior_vectorized(subjects_data, smoothed_stats, Theta_m, Lambda_m)
                loss.backward()
                
                # FIX 2: Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            final_loss = epoch_loss / m_step_iters
            loss_history.append(final_loss)
            
            if verbose and (epoch + 1) % max(1, main_epochs // 5) == 0:
                print(f"    [Main EM] Epoch {epoch+1}/{main_epochs} | Loss: {final_loss:.4f}")
                
        if verbose:
            print(f"  --- EM Complete. Final Loss: {final_loss:.4f} ---\n")
                
        return smoothed_stats, final_loss, loss_history

# ---------------------------------------------------------
# 2. Disease Progression Data Simulation
# ---------------------------------------------------------
def get_true_parameters(D, K, C_dim, theta_mode="dense", seed=42):
    torch.manual_seed(seed)
    
    if theta_mode == "diagonal":
        rho_true = torch.linspace(0.02, 0.15, K)
        Theta_true = torch.diag(rho_true)
    else:
        L_true = torch.tril(torch.randn(K, K) * 0.3 + torch.eye(K)*0.5)
        K_unc = torch.randn(K, K) * 0.2
        Theta_true = L_true @ L_true.T + 1e-4 * torch.eye(K) + K_unc - K_unc.T
        
    B_true, C_true, d_true = torch.randn(K, C_dim)*0.5, torch.randn(K, C_dim)*0.5, torch.randn(K)*0.5
    Z_true = torch.randn(D, K) - 1.0 
    Lambda_true = torch.tril(torch.ones(D, K)) * torch.exp(Z_true)
    
    return {'Lambda': Lambda_true, 'Theta': Theta_true, 'B': B_true, 'C': C_true, 'd': d_true}

def generate_subjects_from_params(N, true_params, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    Lambda_true = true_params['Lambda']
    Theta_true = true_params['Theta']
    B_true = true_params['B']
    C_true = true_params['C']
    d_true = true_params['d']
    
    D, K = Lambda_true.shape
    C_dim = B_true.shape[1]
    
    subjects_data = []
    for _ in range(N):
        J_i = torch.randint(3, 6, (1,)).item()
        age_baseline = torch.rand(1) * 20 + 55
        dt = torch.rand(J_i - 1) * 3.5 + 1.5
        times = torch.cat([age_baseline, age_baseline + torch.cumsum(dt, dim=0)])
        
        t_scaled = (times - 70.0) / 10.0 
        u = torch.randn(J_i, C_dim)
        
        F_true = torch.zeros(J_i, K)
        F_true[0] = torch.randn(K) * 0.1
        
        for j in range(1, J_i):
            delta_t = times[j] - times[j-1]
            A_ij = torch.linalg.matrix_exp(-Theta_true * delta_t)
            mu_j = B_true @ u[j] + C_true @ (u[j] * t_scaled[j]) + d_true
            F_true[j] = A_ij @ F_true[j-1] + ((torch.eye(K) - A_ij) @ mu_j) + (torch.randn(K) * torch.sqrt(delta_t))
            
        X_obs = F_true @ Lambda_true.T + torch.randn(J_i, D)
        subjects_data.append({'x': X_obs, 'u': u, 't': t_scaled, 't_raw': times, 'F_true': F_true})
        
    return subjects_data


# ---------------------------------------------------------
# 3. HPC Parallelized Execution Logic
# ---------------------------------------------------------

SCENARIOS = [
    {"name": "1. Baseline Sparse",     "N": 50,  "D": 20,   "K": 3, "C": 2},
    {"name": "2. High-Dim Proteomics", "N": 100, "D": 200,  "K": 5, "C": 2},
    {"name": "3. Ultra High-Dim",      "N": 100, "D": 1000, "K": 5, "C": 2},
    {"name": "4. Complex Pathways",    "N": 100, "D": 50,   "K": 10,"C": 3},
    {"name": "5. Large Cohort",        "N": 500, "D": 50,   "K": 5, "C": 2},
    {"name": "6. Ultimate High-Dim",   "N": 500, "D": 10000,"K": 20, "C": 2}
]
DATA_MODES = ["diagonal", "dense"]
MODEL_MODES = ["diagonal", "dense"]

def generate_task_list(n_runs):
    """Flattens the nested loops into a linear list of tasks."""
    tasks = []
    for s_idx, s in enumerate(SCENARIOS):
        for d_mode in DATA_MODES:
            for m_mode in MODEL_MODES:
                for run_idx in range(n_runs):
                    tasks.append((s_idx, d_mode, m_mode, run_idx))
    return tasks

def run_single_task(task_id, n_runs, out_dir):
    """Executes exactly one slice of the grid based on task_id."""
    os.makedirs(out_dir, exist_ok=True)
    tasks = generate_task_list(n_runs)
    
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Task ID {task_id} out of bounds. Valid range: 0 to {len(tasks)-1}")
        
    s_idx, d_mode, m_mode, run_idx = tasks[task_id]
    s = SCENARIOS[s_idx]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Task {task_id} ---")
    print(f"Device: {device.type.upper()} | Scenario: {s['name']} | Data: {d_mode} | Model: {m_mode} | Run: {run_idx}")

    # 1. Generate underlying True Parameters
    true_params = get_true_parameters(s["D"], s["K"], s["C"], theta_mode=d_mode, seed=42)
    
    # 2. Generate new subject trajectories
    run_seed = abs(hash(f"{s['name']}_{d_mode}_{m_mode}_{run_idx}")) % (2**32)
    subjects_data_cpu = generate_subjects_from_params(s["N"], true_params, seed=run_seed)
    
    # Send to device
    subjects_data = []
    for subj in subjects_data_cpu:
        subjects_data.append({
            'x': subj['x'].to(device), 'u': subj['u'].to(device), 't': subj['t'].to(device),
            'F_true': subj['F_true']
        })

    model = Universal_DFOULS(obs_dim=s["D"], latent_dim=s["K"], covar_dim=s["C"], theta_mode=m_mode).to(device)
    
    start_time = time.time()
    # Call the multistart fit function
    smoothed_stats, final_loss, loss_history = model.fit_em_multistart(
        subjects_data, 
        num_em_epochs=1000, 
        m_step_iters=20, 
        lr=0.005, 
        n_starts=3, 
        burn_in_epochs=10,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    # Metrics Evaluation
    with torch.no_grad():
        mask_cpu = model.tril_mask.cpu() == 1
        l_true_active = true_params['Lambda'][mask_cpu].numpy()
        th_true_mat = true_params['Theta'].numpy()
        th_true_flat = th_true_mat.flatten()
        b_true_flat = true_params['B'].numpy().flatten()
        c_true_flat = true_params['C'].numpy().flatten()
        f_true_flat = torch.cat([subj['F_true'] for subj in subjects_data], dim=0).numpy().flatten()
        
        Lambda_est_cpu = (model.tril_mask * torch.exp(model.Z)).cpu()
        l_est_active = Lambda_est_cpu[mask_cpu].numpy()
        Theta_est_cpu = model.get_theta().cpu()
        b_est_flat = model.B.detach().cpu().numpy().flatten()
        c_est_flat = model.C_int.detach().cpu().numpy().flatten()
        f_est_flat = torch.cat([stat[0].cpu() for stat in smoothed_stats], dim=0).numpy().flatten()
        
        l_corr = np.corrcoef(l_true_active, l_est_active)[0, 1]
        l_mse = np.mean((l_true_active - l_est_active)**2)
        f_corr = np.corrcoef(f_true_flat, f_est_flat)[0, 1]
        f_mse = np.mean((f_true_flat - f_est_flat)**2)
        
        th_est_mat = np.diag(np.diag(Theta_est_cpu.numpy())) if m_mode == "diagonal" else Theta_est_cpu.numpy()
        th_est_flat = th_est_mat.flatten()
        th_corr = np.corrcoef(th_true_flat, th_est_flat)[0, 1]
        th_mse = np.mean((th_true_flat - th_est_flat)**2)
        off_diag_mask = ~np.eye(s["K"], dtype=bool)
        th_off_mse = np.mean((th_true_mat[off_diag_mask] - th_est_mat[off_diag_mask])**2)
        
        b_corr = np.corrcoef(b_true_flat, b_est_flat)[0, 1]
        b_mse = np.mean((b_true_flat - b_est_flat)**2)
        c_corr = np.corrcoef(c_true_flat, c_est_flat)[0, 1]
        c_mse = np.mean((c_true_flat - c_est_flat)**2)

    # Compile result dictionary
    result = {
        'metadata': {'Scenario': s['name'], 'Data_Mode': d_mode, 'Model_Mode': m_mode, 'Run_Idx': run_idx, 'Time_s': elapsed},
        'metrics': {
            'Final_Loss': final_loss, 'L_Corr': l_corr, 'L_MSE': l_mse, 
            'F_Corr': f_corr, 'F_MSE': f_mse, 'Theta_Corr': th_corr, 
            'Theta_MSE': th_mse, 'Theta_OffDiag_MSE': th_off_mse,
            'B_Corr': b_corr, 'B_MSE': b_mse, 'C_Corr': c_corr, 'C_MSE': c_mse
        },
        'parameters': {
            'true_params': {
                'Lambda': true_params['Lambda'].numpy(), 'Theta': th_true_mat,
                'B': true_params['B'].numpy(), 'C': true_params['C'].numpy()
            },
            'est_params': {
                'Lambda': Lambda_est_cpu.numpy(), 'Theta': th_est_mat,
                'B': model.B.detach().cpu().numpy(), 'C': model.C_int.detach().cpu().numpy()
            },
            'loss_history': loss_history
        }
    }

    # Save Partial Result
    save_path = os.path.join(out_dir, f"result_task_{task_id:04d}.pt")
    torch.save(result, save_path)
    print(f"Task {task_id} complete in {elapsed:.1f}s. Saved to {save_path}")

def merge_results(out_dir):
    """Combines all partial .pt files into the final consolidated CSV and Archive."""
    print(f"Merging results from {out_dir}...")
    file_paths = glob.glob(os.path.join(out_dir, "result_task_*.pt"))
    
    if not file_paths:
        print("No result files found to merge.")
        return

    metrics_log = []
    parameter_archive = {}

    for path in file_paths:
        res = torch.load(path)
        m = res['metadata']
        run_id = f"{m['Scenario']}_Data-{m['Data_Mode']}_Model-{m['Model_Mode']}_Run-{m['Run_Idx']}"
        
        parameter_archive[run_id] = {
            'true_params': res['parameters']['true_params'],
            'est_params': res['parameters']['est_params'],
            'metrics': res['metrics'],
            'loss_history': res['parameters']['loss_history']
        }
        
        flat_metric = {**m, **res['metrics']}
        metrics_log.append(flat_metric)
        
    df_raw = pd.DataFrame(metrics_log)
    
    # Compute Means and Stds automatically by grouping
    group_cols = ['Scenario', 'Data_Mode', 'Model_Mode']
    df_summary = df_raw.groupby(group_cols).agg(['mean', 'std']).reset_index()
    
    # Flatten multi-index columns
    df_summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_summary.columns.values]
    
    # Updated output filenames for the id_multi variant
    df_summary.to_csv("metrics_summary_id_multi.csv", index=False)
    torch.save(parameter_archive, "parameter_archive_id_multi.pt")
    
    print(f"[+] Merged {len(file_paths)} tasks.")
    print("[+] Saved 'metrics_summary_id_multi.csv' and 'parameter_archive_id_multi.pt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPC Parallel DFOULS Benchmarking")
    parser.add_argument("--task_id", type=int, default=-1, help="SLURM Array Task ID (e.g., $SLURM_ARRAY_TASK_ID). If -1, runs sequentially locally.")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs per scenario combination.")
    parser.add_argument("--out_dir", type=str, default="hpc_results", help="Directory for partial results.")
    parser.add_argument("--merge", action="store_true", help="Merge all partial results in out_dir.")
    args = parser.parse_args()

    if args.merge:
        merge_results(args.out_dir)
    elif args.task_id != -1:
        run_single_task(args.task_id, args.n_runs, args.out_dir)
    else:
        print("No task_id specified. Running all tasks sequentially...")
        total_tasks = len(generate_task_list(args.n_runs))
        for t_id in range(total_tasks):
            run_single_task(t_id, args.n_runs, args.out_dir)
        merge_results(args.out_dir)