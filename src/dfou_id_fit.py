import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import os

# ---------------------------------------------------------
# 1. GPU-Accelerated Universal DFOULS Model
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
        
        R_mat = torch.eye(self.D, device=device)
        I_k = torch.eye(self.K, device=device)
        
        for j in range(1, T):
            idx = j - 1
            f_pred[j] = A_trans[idx] @ f_filt[j-1] + b_shift[idx]
            P_pred[j] = A_trans[idx] @ P_filt[j-1] @ A_trans[idx].T + (dt[idx] * I_k)
            
            if torch.isnan(x_obs[j]).all():
                f_filt[j], P_filt[j] = f_pred[j], P_pred[j]
            else:
                x_pred = Lambda @ f_pred[j]
                S_t = Lambda @ P_pred[j] @ Lambda.T + R_mat
                K_gain = P_pred[j] @ Lambda.T @ torch.linalg.inv(S_t)
                f_filt[j] = f_pred[j] + K_gain @ (x_obs[j] - x_pred)
                P_filt[j] = (I_k - K_gain @ Lambda) @ P_pred[j]
            
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

    def fit_em(self, subjects_data, num_em_epochs=40, m_step_iters=20, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        final_loss = 0.0
        for epoch in range(num_em_epochs):
            Theta, Lambda = self.get_theta(), self.tril_mask * torch.exp(self.Z)
            
            smoothed_stats = []
            with torch.no_grad():
                for subj in subjects_data:
                    A_trans, b_shift, dt, _ = self.get_subject_matrices(Theta, subj['u'], subj['t'])
                    smoothed_stats.append(self.kalman_smoother(subj['x'], A_trans, b_shift, dt, Lambda))
            
            epoch_loss = 0.0
            # Generalized EM Step (Adam handles Z alongside dynamic parameters seamlessly)
            for m in range(m_step_iters):
                optimizer.zero_grad()
                Theta_m, Lambda_m = self.get_theta(), self.tril_mask * torch.exp(self.Z)
                loss = -self.expected_complete_log_posterior_vectorized(subjects_data, smoothed_stats, Theta_m, Lambda_m)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / m_step_iters
        return smoothed_stats, final_loss

# ---------------------------------------------------------
# 2. Disease Progression Data Simulation
# ---------------------------------------------------------
def simulate_ad_cohort_stress(N, D, K, C_dim, theta_mode="dense", seed=42):
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
        
    true_params = {
        'Lambda': Lambda_true, 
        'Theta': Theta_true,
        'B': B_true,
        'C': C_true,
        'd': d_true,
        'F': F_true
    }
    return subjects_data, true_params

# ---------------------------------------------------------
# 3. Robust GPU Benchmarking Execution (Expanded Metrics)
# ---------------------------------------------------------
def run_misspecification_test(n_runs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing Misspecification Benchmarks on: {device.type.upper()}\n")
    
    scenarios = [
        {"name": "1. Baseline Sparse",     "N": 50,  "D": 20,   "K": 3, "C": 2},
        {"name": "2. High-Dim Proteomics", "N": 100, "D": 200,  "K": 5, "C": 2},
        {"name": "3. Ultra High-Dim",      "N": 100, "D": 1000, "K": 5, "C": 2},
        # Restored Scenarios:
        {"name": "4. Complex Pathways",    "N": 100, "D": 50,   "K": 10,"C": 3},
        {"name": "5. Large Cohort",        "N": 500, "D": 50,   "K": 5, "C": 2}
    ]
    
    data_modes = ["diagonal", "dense"]
    model_modes = ["diagonal", "dense"]
    
    print(f"{'Scenario':<22} | {'Data':<8} | {'Model':<8} | {'Λ Corr':<8} | {'F Corr':<8} | {'Θ Corr':<8} | {'Θ MSE':<8} | {'Time (s)'}")
    print("-" * 92)
    
    metrics_log = []
    parameter_archive = {}
    
    for s in scenarios:
        for d_mode in data_modes:
            subjects_data_cpu, true_params = simulate_ad_cohort_stress(
                s["N"], s["D"], s["K"], s["C"], theta_mode=d_mode, seed=42
            )
            
            subjects_data = []
            for subj in subjects_data_cpu:
                subjects_data.append({
                    'x': subj['x'].to(device),
                    'u': subj['u'].to(device),
                    't': subj['t'].to(device),
                    'F_true': subj['F_true']
                })
                
            for m_mode in model_modes:
                # Metric Accumulators
                l_corrs, l_mses = [], []
                f_corrs, f_mses = [], []
                th_corrs, th_mses, th_off_mses = [], [], []
                b_corrs, b_mses = [], []
                c_corrs, c_mses = [], []
                final_losses, run_times = [], []
                
                for run_idx in range(n_runs):
                    start_time = time.time()
                    
                    model = Universal_DFOULS(obs_dim=s["D"], latent_dim=s["K"], covar_dim=s["C"], theta_mode=m_mode).to(device)
                    smoothed_stats, final_loss = model.fit_em(
                        subjects_data, num_em_epochs=40, m_step_iters=20, lr=0.01)
                    
                    with torch.no_grad():
                        # Extract True Parameters
                        mask_cpu = model.tril_mask.cpu() == 1
                        l_true_active = true_params['Lambda'][mask_cpu].numpy()
                        th_true_mat = true_params['Theta'].numpy()
                        th_true_flat = th_true_mat.flatten()
                        b_true_flat = true_params['B'].numpy().flatten()
                        c_true_flat = true_params['C'].numpy().flatten()
                        f_true_flat = torch.cat([subj['F_true'] for subj in subjects_data], dim=0).numpy().flatten()
                        
                        # Extract Estimated Parameters
                        Lambda_est_cpu = (model.tril_mask * torch.exp(model.Z)).cpu()
                        l_est_active = Lambda_est_cpu[mask_cpu].numpy()
                        Theta_est_cpu = model.get_theta().cpu()
                        b_est_flat = model.B.detach().cpu().numpy().flatten()
                        c_est_flat = model.C_int.detach().cpu().numpy().flatten()
                        f_est_flat = torch.cat([stat[0].cpu() for stat in smoothed_stats], dim=0).numpy().flatten()
                        
                        # Lambda & F Metrics
                        l_corr = np.corrcoef(l_true_active, l_est_active)[0, 1]
                        l_mse = np.mean((l_true_active - l_est_active)**2)
                        
                        f_corr = np.corrcoef(f_true_flat, f_est_flat)[0, 1]
                        f_mse = np.mean((f_true_flat - f_est_flat)**2)
                        
                        # Theta Misspecification Matrices
                        if m_mode == "diagonal":
                            th_est_mat = np.diag(np.diag(Theta_est_cpu.numpy()))
                        else:
                            th_est_mat = Theta_est_cpu.numpy()
                        th_est_flat = th_est_mat.flatten()
                            
                        # Theta Metrics
                        th_corr = np.corrcoef(th_true_flat, th_est_flat)[0, 1]
                        th_mse = np.mean((th_true_flat - th_est_flat)**2)
                        
                        # Off-Diagonal specific MSE (Critical for testing interaction hallucination)
                        off_diag_mask = ~np.eye(s["K"], dtype=bool)
                        th_off_mse = np.mean((th_true_mat[off_diag_mask] - th_est_mat[off_diag_mask])**2)
                        
                        # Covariate Metrics
                        b_corr = np.corrcoef(b_true_flat, b_est_flat)[0, 1]
                        b_mse = np.mean((b_true_flat - b_est_flat)**2)
                        c_corr = np.corrcoef(c_true_flat, c_est_flat)[0, 1]
                        c_mse = np.mean((c_true_flat - c_est_flat)**2)
                    
                    elapsed = time.time() - start_time
                    
                    # Accumulate for Averaging
                    l_corrs.append(l_corr); l_mses.append(l_mse)
                    f_corrs.append(f_corr); f_mses.append(f_mse)
                    th_corrs.append(th_corr); th_mses.append(th_mse); th_off_mses.append(th_off_mse)
                    b_corrs.append(b_corr); b_mses.append(b_mse)
                    c_corrs.append(c_corr); c_mses.append(c_mse)
                    final_losses.append(final_loss)
                    run_times.append(elapsed)
                    
                    # Archive exact state dictionaries
                    run_id = f"{s['name']}_Data-{d_mode}_Model-{m_mode}_Run-{run_idx}"
                    parameter_archive[run_id] = {
                        'true_params': {
                            'Lambda': true_params['Lambda'].numpy(),
                            'Theta': th_true_mat,
                            'B': true_params['B'].numpy(),
                            'C': true_params['C'].numpy()
                        },
                        'est_params': {
                            'Lambda': Lambda_est_cpu.numpy(),
                            'Theta': th_est_mat,
                            'B': model.B.detach().cpu().numpy(),
                            'C': model.C_int.detach().cpu().numpy()
                        },
                        'metrics': {'Final_Loss': final_loss, 'L_mse': l_mse, 'F_mse': f_mse, 'Theta_mse': th_mse}
                    }
                
                # Consolidate Console Print (Keeping it clean for terminal)
                l_mu = np.mean(l_corrs)
                f_mu = np.mean(f_corrs)
                th_c_mu = np.mean(th_corrs)
                th_m_mu = np.mean(th_mses)
                time_avg = np.mean(run_times)
                
                print(f"{s['name']:<22} | {d_mode.capitalize():<8} | {m_mode.capitalize():<8} | {l_mu:>6.3f}   | {f_mu:>6.3f}   | {th_c_mu:>6.3f}   | {th_m_mu:>6.3f}   | {time_avg:>8.1f}")
                
                # Full logging to Pandas DataFrame
                metrics_log.append({
                    'Scenario': s['name'], 'Data_Mode': d_mode, 'Model_Mode': m_mode,
                    'Final_Loss_Mean': np.mean(final_losses), 'Final_Loss_Std': np.std(final_losses),
                    'L_Corr_Mean': l_mu, 'L_Corr_Std': np.std(l_corrs),
                    'L_MSE_Mean': np.mean(l_mses), 'L_MSE_Std': np.std(l_mses),
                    'F_Corr_Mean': f_mu, 'F_Corr_Std': np.std(f_corrs),
                    'F_MSE_Mean': np.mean(f_mses), 'F_MSE_Std': np.std(f_mses),
                    'Theta_Corr_Mean': th_c_mu, 'Theta_Corr_Std': np.std(th_corrs),
                    'Theta_MSE_Mean': th_m_mu, 'Theta_MSE_Std': np.std(th_mses),
                    'Theta_OffDiag_MSE_Mean': np.mean(th_off_mses), 'Theta_OffDiag_MSE_Std': np.std(th_off_mses),
                    'B_Corr_Mean': np.mean(b_corrs), 'B_MSE_Mean': np.mean(b_mses),
                    'C_Corr_Mean': np.mean(c_corrs), 'C_MSE_Mean': np.mean(c_mses),
                    'Avg_Time_s': time_avg
                })
        print("-" * 92)

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv("metrics_summary_id_fit.csv", index=False)
    torch.save(parameter_archive, "parameter_archive_id_fit.pt")
    print("\n[+] Saved extensive metrics to 'metrics_summary_id_fit.csv' and matrices to 'parameter_archive_id_fit.pt'")

if __name__ == "__main__":
    run_misspecification_test(n_runs=10)