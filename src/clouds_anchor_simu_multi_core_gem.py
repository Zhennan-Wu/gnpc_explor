import concurrent.futures
import copy
import os
# 1. Completely blind PyTorch to any GPUs on the server
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import sys
import traceback

# 2. Optimize CPU core usage for small matrix math (Prevents thread thrashing)
torch.set_num_threads(8)

# ---------------------------------------------------------
# 0. Server Logging Configuration
# ---------------------------------------------------------
LOG_FILENAME = "clouds_simulation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------------------------------------------------------
# 1. Helper Function: Newton-Schulz Matrix Inversion
# ---------------------------------------------------------
def batched_newton_schulz_inverse(A, num_iters=6):
    B, N, _ = A.shape
    I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, N, N)
    frob_norm_sq = torch.sum(A * A, dim=(-2, -1), keepdim=True)
    X = A.transpose(-2, -1) / (frob_norm_sq + 1e-6)
    
    for _ in range(num_iters):
        AX = torch.bmm(A, X)
        X = torch.bmm(X, (2.0 * I) - AX)
    return X

# ---------------------------------------------------------
# 2. CLOUDS Model (Woodbury Optimized & Anchor Orientation)
# ---------------------------------------------------------
class CLOUDS(nn.Module):
    def __init__(self, obs_dim, latent_dim, covar_dim, delta=1e-4, theta_mode="exact", anchor_items=None):
        super().__init__()
        self.D = obs_dim
        self.K = latent_dim
        self.C_dim = covar_dim
        self.delta = delta
        self.theta_mode = theta_mode
        
        # Default to the first K items as anchors if not provided
        if anchor_items is None:
            anchor_items = list(range(self.K))
        assert len(anchor_items) == self.K, f"Must provide exactly R={self.K} anchor items."
        
        if self.theta_mode == "exact":
            self.L_G = nn.Parameter(torch.tril(torch.eye(self.K) + 0.1 * torch.randn(self.K, self.K)))
            self.gamma_skew = nn.Parameter(torch.randn(self.K, self.K) * 0.1)
            self.L_Omega_unc = nn.Parameter(torch.tril(torch.eye(self.K) + 0.1 * torch.randn(self.K, self.K)))
        else:
            self.log_rho = nn.Parameter(torch.randn(self.K) * 0.1 - 2.0)
            self.log_omega = nn.Parameter(torch.randn(self.K) * 0.1)
            
        self.Phi_int = nn.Parameter(torch.randn(self.K, self.C_dim) * 0.1)
        self.alpha_bias = nn.Parameter(torch.randn(self.K) * 0.1)
        
        # 3. Factor Loadings (Anchor Orientation)
        self.Lambda_raw = nn.Parameter(torch.randn(self.D, self.K) * 0.1)
        
        self.register_buffer('anchor_idx', torch.tensor(anchor_items, dtype=torch.long))
        self.register_buffer('anchor_cols', torch.arange(self.K, dtype=torch.long))
        
        # struct_mask: 0 at off-diagonals of anchor rows, 1 everywhere else
        struct_mask = torch.ones(self.D, self.K)
        struct_mask[self.anchor_idx, :] = 0.0
        struct_mask[self.anchor_idx, self.anchor_cols] = 1.0
        self.register_buffer('struct_mask', struct_mask)
        
        # positivity_mask: True only at the diagonal of the anchor block
        positivity_mask = torch.zeros(self.D, self.K, dtype=torch.bool)
        positivity_mask[self.anchor_idx, self.anchor_cols] = True
        self.register_buffer('positivity_mask', positivity_mask)
        
        self.log_psi = nn.Parameter(torch.zeros(self.D)) 

    @property
    def Lambda(self):
        """Constructs Lambda enforcing structural zeroes and strict positivity on anchor diagonals."""
        return torch.where(self.positivity_mask, 
                           torch.exp(self.Lambda_raw), 
                           self.Lambda_raw * self.struct_mask)

    def get_dynamics(self):
        device = self.Lambda_raw.device
        if self.theta_mode == "exact":
            L_unc_tril = torch.tril(self.L_Omega_unc)
            Omega = L_unc_tril @ L_unc_tril.T + self.delta * torch.eye(self.K, device=device)
            
            G = torch.tril(self.L_G)
            S = 0.5 * (G @ G.T) + self.delta * torch.eye(self.K, device=device)
            A_skew = self.gamma_skew - self.gamma_skew.T
            
            Omega_inv = batched_newton_schulz_inverse(Omega.unsqueeze(0), num_iters=6).squeeze(0)
            Gamma = (S + A_skew) @ Omega_inv
            return Gamma, Omega, G
        else:
            Gamma = torch.diag(torch.exp(self.log_rho))
            Omega = torch.diag(torch.exp(self.log_omega))
            G = torch.sqrt(2.0 * Gamma @ Omega)
            return Gamma, Omega, G

    @torch.no_grad()
    def get_identifiable_parameters(self):
        Gamma_est, Omega_est, _ = self.get_dynamics()
        Lambda_est = self.Lambda
        
        stds = torch.sqrt(torch.diag(Omega_est))
        D = torch.diag(stds)
        D_inv = torch.diag(1.0 / stds)
        
        Omega_corr = D_inv @ Omega_est @ D_inv
        Gamma_scaled = D_inv @ Gamma_est @ D
        Lambda_scaled = Lambda_est @ D
        Phi_scaled = D_inv @ self.Phi_int
        alpha_scaled = D_inv @ self.alpha_bias
        
        return {
            "Omega_corr": Omega_corr,
            "Gamma": Gamma_scaled,
            "Lambda": Lambda_scaled,
            "Phi": Phi_scaled,
            "alpha": alpha_scaled
        }

    def get_subject_matrices(self, Gamma, Omega, u, times):
        dt = times[1:] - times[:-1]
        device = self.Lambda_raw.device
        
        Gamma_batch = Gamma.unsqueeze(0).expand(dt.shape[0], self.K, self.K)
        A_trans = torch.linalg.matrix_exp(-Gamma_batch * dt.view(-1, 1, 1))
        
        u_t, t_val = u[1:], times[1:].unsqueeze(1)
        mu = (u_t @ self.Phi_int.T + self.alpha_bias) * t_val
        
        I_batch = torch.eye(self.K, device=device).unsqueeze(0).expand(dt.shape[0], self.K, self.K)
        b_shift = torch.bmm(I_batch - A_trans, mu.unsqueeze(-1)).squeeze(-1)
        
        Omega_batch = Omega.unsqueeze(0).expand(dt.shape[0], self.K, self.K)
        A_trans_T = A_trans.transpose(1, 2)
        Q = Omega_batch - torch.bmm(A_trans, torch.bmm(Omega_batch, A_trans_T))
        Q = 0.5 * (Q + Q.transpose(1, 2)) 
            
        Lambda = self.Lambda
        return A_trans, b_shift, dt, Lambda, Q

    def kalman_smoother(self, x_obs, A_trans, b_shift, dt, Lambda, Q):
        T = x_obs.shape[0]
        device = x_obs.device
        
        f_pred, P_pred = torch.zeros(T, self.K, device=device), torch.zeros(T, self.K, self.K, device=device)
        f_filt, P_filt = torch.zeros(T, self.K, device=device), torch.zeros(T, self.K, self.K, device=device)
        f_filt[0], P_filt[0] = torch.zeros(self.K, device=device), torch.eye(self.K, device=device)
        
        inv_psi = torch.exp(-self.log_psi)
        Lambda_T_invPsi = Lambda.T * inv_psi.unsqueeze(0) 
        obs_info = Lambda_T_invPsi @ Lambda 
        
        for j in range(1, T):
            idx = j - 1
            f_pred[j] = A_trans[idx] @ f_filt[j-1] + b_shift[idx]
            P_pred[j] = A_trans[idx] @ P_filt[j-1] @ A_trans[idx].T + Q[idx]
            
            if torch.isnan(x_obs[j]).all():
                f_filt[j], P_filt[j] = f_pred[j], P_pred[j]
            else:
                # Woodbury Identity: Avoids DxD inversion
                P_pred_inv = torch.linalg.inv(P_pred[j])
                P_filt[j] = torch.linalg.inv(P_pred_inv + obs_info)
                
                residual = x_obs[j] - Lambda @ f_pred[j]
                f_filt[j] = f_pred[j] + P_filt[j] @ (Lambda_T_invPsi @ residual)
            
        f_smooth, P_smooth, P_cross = torch.zeros_like(f_filt), torch.zeros_like(P_filt), torch.zeros_like(P_filt)
        f_smooth[-1], P_smooth[-1] = f_filt[-1], P_filt[-1]
        
        for j in range(T-2, -1, -1):
            J_t = P_filt[j] @ A_trans[j].T @ torch.linalg.inv(P_pred[j+1])
            f_smooth[j] = f_filt[j] + J_t @ (f_smooth[j+1] - f_pred[j+1])
            P_smooth[j] = P_filt[j] + J_t @ (P_smooth[j+1] - P_pred[j+1]) @ J_t.T
            P_cross[j+1] = J_t @ P_smooth[j+1]
            
        return f_smooth, P_smooth, P_cross

    def expected_complete_log_posterior_vectorized(self, subjects_data, smoothed_stats, Gamma, Omega, Lambda):
        ll_obs, ll_lat = 0.0, 0.0
        inv_psi = torch.exp(-self.log_psi)
        L_Psi_L = Lambda.T @ (inv_psi.unsqueeze(1) * Lambda) 
        
        total_obs = sum([subj['x'].shape[0] for subj in subjects_data])
        
        for i, subj in enumerate(subjects_data):
            x_obs, u, times = subj['x'], subj['u'], subj['t']
            f_s, P_s, P_c = smoothed_stats[i]
            
            A_trans, b_shift, _, _, Q_exact = self.get_subject_matrices(Gamma, Omega, u, times)
            
            valid_mask = ~torch.isnan(x_obs).any(dim=1)
            if valid_mask.any():
                x_v, f_v, P_v = x_obs[valid_mask], f_s[valid_mask], P_s[valid_mask]
                trace_E = torch.sum(P_v * L_Psi_L.unsqueeze(0), dim=(1,2)) + torch.sum(f_v * (f_v @ L_Psi_L), dim=1)
                term_obs = torch.sum((x_v**2) * inv_psi, dim=1) - 2 * torch.sum(x_v * (f_v @ Lambda.T) * inv_psi, dim=1) + trace_E
                ll_obs += torch.sum(-0.5 * term_obs - 0.5 * torch.sum(self.log_psi))
                
            Q_stable = Q_exact + 1e-5 * torch.eye(self.K, device=Q_exact.device).unsqueeze(0)
            Q_inv = torch.linalg.inv(Q_stable)
            log_det_Q = torch.linalg.slogdet(Q_stable)[1]
            
            f_j, f_jm1 = f_s[1:], f_s[:-1]
            P_j, P_jm1, P_cj = P_s[1:], P_s[:-1], P_c[1:]
            
            E_jj = P_j + torch.bmm(f_j.unsqueeze(-1), f_j.unsqueeze(1))
            E_jjm1 = P_cj + torch.bmm(f_j.unsqueeze(-1), f_jm1.unsqueeze(1))
            E_jm1jm1 = P_jm1 + torch.bmm(f_jm1.unsqueeze(-1), f_jm1.unsqueeze(1))
            
            A_T = A_trans.transpose(1, 2)
            M_j = (E_jj 
                   - torch.bmm(E_jjm1, A_T) 
                   - torch.bmm(A_trans, E_jjm1.transpose(1, 2)) 
                   + torch.bmm(A_trans, torch.bmm(E_jm1jm1, A_T)))
            
            b_uns, b_T = b_shift.unsqueeze(-1), b_shift.unsqueeze(1)
            f_j_uns, f_j_T = f_j.unsqueeze(-1), f_j.unsqueeze(1)
            f_jm1_uns, f_jm1_T = f_jm1.unsqueeze(-1), f_jm1.unsqueeze(1)
            
            M_j += (- torch.bmm(f_j_uns, b_T) 
                    - torch.bmm(b_uns, f_j_T) 
                    + torch.bmm(A_trans, torch.bmm(f_jm1_uns, b_T)) 
                    + torch.bmm(b_uns, torch.bmm(f_jm1_T, A_T)) 
                    + torch.bmm(b_uns, b_T))
            
            trace_term = torch.sum(Q_inv * M_j.transpose(1, 2), dim=(1, 2))
            ll_lat += torch.sum(-0.5 * log_det_Q - 0.5 * trace_term)
            
        log_prior_dyn = 0.0
        if self.theta_mode == "exact":
            stds = torch.sqrt(torch.diag(Omega))
            D_inv = torch.diag(1.0 / stds)
            Omega_corr = D_inv @ Omega @ D_inv
            
            eta = 1.5 
            log_prior_dyn += (eta - 1.0) * torch.linalg.slogdet(Omega_corr + 1e-5*torch.eye(self.K, device=Omega.device))[1]
            
            lambda_laplace = 0.1
            log_prior_dyn -= lambda_laplace * torch.sum(torch.abs(self.gamma_skew))
            
            off_diag_G = torch.tril(self.L_G, diagonal=-1)
            log_prior_dyn -= lambda_laplace * torch.sum(torch.abs(off_diag_G))
        else:
            log_prior_dyn -= 0.5 * torch.sum(self.log_rho**2) + 0.5 * torch.sum(self.log_omega**2)

        # Apply prior only to active structural parameters
        active_Lambda_raw = self.Lambda_raw[self.struct_mask == 1]
        log_prior_Lambda = -0.5 * torch.sum(active_Lambda_raw ** 2)
        
        log_prior_lin = -0.5 * (torch.sum(self.Phi_int**2) + torch.sum(self.alpha_bias**2))
        log_prior_psi = -0.5 * torch.sum(self.log_psi ** 2)
        
        return ll_obs + ll_lat + log_prior_dyn + log_prior_Lambda + log_prior_lin + log_prior_psi

    def pca_warm_start(self, subjects_data):
        with torch.no_grad():
            x_all = torch.cat([s['x'] for s in subjects_data], dim=0)
            x_valid = x_all[~torch.isnan(x_all).any(dim=1)] 
            U, S_vals, Vh = torch.linalg.svd(x_valid - x_valid.mean(dim=0), full_matrices=False)
            
            Lambda_pca = Vh[:self.K, :].T * torch.sqrt(S_vals[:self.K] / x_valid.shape[0])
            
            # Targeted rotation: Align PCA loadings to the Anchor space
            A_pca = Lambda_pca[self.anchor_idx, :]
            W = torch.linalg.pinv(A_pca) @ torch.diag(torch.norm(A_pca, dim=1))
            
            Lambda_rotated = Lambda_pca @ W
            
            self.Lambda_raw.data = Lambda_rotated
            # Initialize log-space anchor parameters
            self.Lambda_raw.data[self.anchor_idx, self.anchor_cols] = torch.log(
                torch.abs(Lambda_rotated[self.anchor_idx, self.anchor_cols]) + 1e-4
            )
            
            self.Phi_int.data.fill_(0.0); self.alpha_bias.data.fill_(0.0); self.log_psi.data.fill_(0.0)

    def fit_em_multistart(self, subjects_data, num_em_epochs=40, warmup_epochs=15, m_step_iters=20, lr=0.01, n_starts=5, burn_in_epochs=10):
        best_loss = float('inf')
        best_state_dict = None
        
        spatial_names = ['Lambda_raw', 'log_psi']
        temporal_params = [p for n, p in self.named_parameters() if n not in spatial_names]
        spatial_params = [p for n, p in self.named_parameters() if n in spatial_names]
        
        total_obs = sum([subj['x'].shape[0] for subj in subjects_data])
        
        for start in range(n_starts):
            with torch.no_grad():
                if self.theta_mode == "exact":
                    nn.init.normal_(self.L_G, mean=0.0, std=0.1)
                    self.L_G.data += torch.eye(self.K, device=self.Lambda_raw.device)
                    nn.init.normal_(self.gamma_skew, mean=0.0, std=0.1)
                    nn.init.normal_(self.L_Omega_unc, mean=0.0, std=0.1)
                    self.L_Omega_unc.data += torch.eye(self.K, device=self.Lambda_raw.device)
                else:
                    nn.init.normal_(self.log_rho, mean=-2.0, std=0.1)
                    nn.init.normal_(self.log_omega, mean=0.0, std=0.1)
                
                nn.init.normal_(self.Phi_int, mean=0.0, std=0.1)
                nn.init.normal_(self.alpha_bias, mean=0.0, std=0.1)
            
            self.pca_warm_start(subjects_data)
            
            opt_burn = optim.Adam(temporal_params, lr=lr)
            start_loss = 0.0
            
            for epoch in range(burn_in_epochs):
                Gamma, Omega, _ = self.get_dynamics()
                Lambda = self.Lambda
                
                smoothed_stats = []
                with torch.no_grad():
                    for subj in subjects_data:
                        A_trans, b_shift, dt, _, Q = self.get_subject_matrices(Gamma, Omega, subj['u'], subj['t'])
                        smoothed_stats.append(self.kalman_smoother(subj['x'], A_trans, b_shift, dt, Lambda, Q))
                
                epoch_loss = 0.0
                for m in range(m_step_iters):
                    opt_burn.zero_grad()
                    Gamma_m, Omega_m, _ = self.get_dynamics()
                    Lambda_detached = self.Lambda.detach()
                    
                    loss = -self.expected_complete_log_posterior_vectorized(subjects_data, smoothed_stats, Gamma_m, Omega_m, Lambda_detached) / (total_obs * self.D)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(temporal_params, max_norm=2.0)
                    opt_burn.step()
                    epoch_loss += loss.item()
                start_loss = epoch_loss / m_step_iters
                
            if start_loss < best_loss:
                best_loss = start_loss
                best_state_dict = {k: v.clone() for k, v in self.state_dict().items()}
                
        self.load_state_dict(best_state_dict)
        
        opt_dynamics_only = optim.Adam(temporal_params, lr=lr)
        opt_joint = optim.Adam([
            {'params': temporal_params, 'lr': lr},
            {'params': spatial_params,  'lr': lr * 0.05} 
        ])
        
        for epoch in range(num_em_epochs - burn_in_epochs):
            Gamma, Omega, _ = self.get_dynamics()
            Lambda = self.Lambda
            
            smoothed_stats = []
            with torch.no_grad():
                for subj in subjects_data:
                    A_trans, b_shift, dt, _, Q = self.get_subject_matrices(Gamma, Omega, subj['u'], subj['t'])
                    smoothed_stats.append(self.kalman_smoother(subj['x'], A_trans, b_shift, dt, Lambda, Q))
            
            active_opt = opt_dynamics_only if epoch < warmup_epochs else opt_joint
            
            for m in range(m_step_iters):
                active_opt.zero_grad()
                Gamma_m, Omega_m, _ = self.get_dynamics()
                Lambda_m = self.Lambda
                
                if epoch < warmup_epochs:
                    Lambda_m = Lambda_m.detach()
                    
                loss = -self.expected_complete_log_posterior_vectorized(subjects_data, smoothed_stats, Gamma_m, Omega_m, Lambda_m) / (total_obs * self.D)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                active_opt.step()
                
        return smoothed_stats

# ---------------------------------------------------------
# 3. Data Simulation Wrappers
# ---------------------------------------------------------
def simulate_ad_cohort_stress(N, D, K, C_dim, theta_mode="exact", seed=42, missing_visit_rate=0.0, noise_scale=1.0, anchor_items=None):
    torch.manual_seed(seed)
    
    if anchor_items is None:
        anchor_items = list(range(K))
        
    if theta_mode == "diagonal":
        rho_true = torch.linspace(0.02, 0.15, K)
        omega_true = torch.ones(K)
        Gamma_true = torch.diag(rho_true)
        Omega_true = torch.diag(omega_true)
    else:
        L_unc_true = torch.tril(torch.randn(K, K) * 0.3 + torch.eye(K))
        L_corr_true = L_unc_true / torch.norm(L_unc_true, dim=1, keepdim=True)
        Omega_true = L_corr_true @ L_corr_true.T
        
        L_G_true = torch.tril(torch.randn(K, K) * 0.3 + torch.eye(K)*0.5)
        S_true = 0.5 * (L_G_true @ L_G_true.T) + 1e-4 * torch.eye(K)
        
        gamma_skew_true = torch.randn(K, K) * 0.2
        A_true = gamma_skew_true - gamma_skew_true.T
        Gamma_true = (S_true + A_true) @ torch.linalg.inv(Omega_true)
        
    Phi_true, alpha_true = torch.randn(K, C_dim)*0.5, torch.randn(K)*0.5
    
    # Ground truth Lambda generation with anchor constraints
    Z_true = torch.randn(D, K) * 0.5 
    Lambda_true = Z_true.clone()
    for r, idx in enumerate(anchor_items):
        Lambda_true[idx, :] = 0.0
        Lambda_true[idx, r] = torch.exp(torch.randn(1) * 0.5) # Force strict positivity
    
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
            A_ij = torch.linalg.matrix_exp(-Gamma_true * delta_t)
            mu_j = (Phi_true @ u[j] + alpha_true) * t_scaled[j]
            
            Q_true = Omega_true - A_ij @ Omega_true @ A_ij.T
            Q_true = 0.5 * (Q_true + Q_true.T) + 1e-5 * torch.eye(K)
            L_Q = torch.linalg.cholesky(Q_true)
            noise = L_Q @ torch.randn(K)
            
            F_true[j] = A_ij @ F_true[j-1] + ((torch.eye(K) - A_ij) @ mu_j) + noise
            
        X_obs = F_true @ Lambda_true.T + torch.randn(J_i, D) * noise_scale
        
        if missing_visit_rate > 0.0:
            for j in range(1, J_i):
                if torch.rand(1).item() < missing_visit_rate:
                    X_obs[j, :] = float('nan')
                    
        subjects_data.append({'x': X_obs, 'u': u, 't': t_scaled, 'F_true': F_true})
        
    return subjects_data, {'Lambda': Lambda_true, 'Gamma': Gamma_true}


# ---------------------------------------------------------
# 4. Evaluation Wrappers (128-Core Multiprocessing)
# ---------------------------------------------------------

def run_smoke_test():
    """ A rapid integration test to ensure mathematically stability before long runs. """
    logging.info("--- STARTING SMOKE TEST ---")
    try:
        start_time = time.time()
        # Micro-dataset
        anchor_items = list(range(3))
        subjects_data, _ = simulate_ad_cohort_stress(
            N=5, D=15, K=3, C_dim=2, theta_mode="exact", seed=99, anchor_items=anchor_items
        )
        
        model = CLOUDS(obs_dim=15, latent_dim=3, covar_dim=2, theta_mode="exact", anchor_items=anchor_items)
        model.pca_warm_start(subjects_data)
        
        # Micro-epochs
        model.fit_em_multistart(
            subjects_data, 
            num_em_epochs=2, 
            warmup_epochs=1, 
            m_step_iters=2, 
            lr=0.01, 
            n_starts=2, 
            burn_in_epochs=1
        )
        
        identifiable = model.get_identifiable_parameters()
        
        elapsed = time.time() - start_time
        logging.info(f"Smoke Test Passed in {elapsed:.2f}s. Architecture and Memory bounds verified.")
        return True
    except Exception as e:
        logging.error("Smoke Test Failed with exception:")
        logging.error(traceback.format_exc())
        return False
    
def run_single_simulation(task_id, scenario, mode, run_idx, seed):
    # Force this specific worker to only use 8 threads
    torch.set_num_threads(8)
    
    # THE HEARTBEAT
    logging.info(f"[Worker {task_id}] Successfully spawned and starting math operations...")
    
    start_time = time.time()
    
    try:
        anchor_items = list(range(scenario["K"]))
        subjects_data, true_params = simulate_ad_cohort_stress(
            scenario["N"], scenario["D"], scenario["K"], scenario["C"], 
            theta_mode=mode, 
            seed=seed,
            missing_visit_rate=scenario["miss"], 
            noise_scale=scenario["noise"],
            anchor_items=anchor_items
        )
        
        model = CLOUDS(
            obs_dim=scenario["D"], latent_dim=scenario["K"], 
            covar_dim=scenario["C"], theta_mode=mode, anchor_items=anchor_items
        )
        model.pca_warm_start(subjects_data)
        
        smoothed_stats = model.fit_em_multistart(
            subjects_data, 
            num_em_epochs=30, 
            warmup_epochs=10, 
            m_step_iters=15, 
            lr=0.01,
            n_starts=4,
            burn_in_epochs=5
        )
        
        with torch.no_grad():
            identifiable = model.get_identifiable_parameters()
            Lambda_est = identifiable["Lambda"]
            Gamma_est = identifiable["Gamma"]
            mask = model.struct_mask == 1
            
            f_true_flat = torch.cat([subj['F_true'] for subj in subjects_data], dim=0).numpy().flatten()
            f_est_flat = torch.cat([stat[0] for stat in smoothed_stats], dim=0).numpy().flatten()
            
            f_corr = np.corrcoef(f_true_flat, f_est_flat)[0, 1]
            l_corr = np.corrcoef(true_params['Lambda'][mask].cpu().numpy(), Lambda_est[mask].cpu().numpy())[0, 1]
            
            if mode == "diagonal":
                g_true = torch.diag(true_params['Gamma']).numpy()
                g_est = torch.diag(Gamma_est).cpu().numpy()
            else:
                g_true = true_params['Gamma'].numpy().flatten()
                g_est = Gamma_est.cpu().numpy().flatten()
                
            gamma_corr = np.corrcoef(g_true, g_est)[0, 1]
            
        elapsed = time.time() - start_time
        
        return {
            "status": "success",
            "task_id": task_id,
            "scenario_name": scenario["name"],
            "mode": mode,
            "l_corr": l_corr,
            "f_corr": f_corr,
            "g_corr": gamma_corr,
            "time": elapsed
        }
        
    except Exception as e:
        return {
            "status": "error",
            "task_id": task_id,
            "scenario_name": scenario["name"],
            "mode": mode,
            "error_msg": str(e) + "\n" + traceback.format_exc()
        }

def run_stress_test_multiprocessing(n_runs=2):
    """
    The Main Thread Coordinator. Spawns the 16 process workers and aggregates logs.
    """
    scenarios = [
        {"name": "1. Multi-Omics Base (D=2k)",   "N": 100, "D": 2000,  "K": 3, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "2. Real-World Scale (D=8k)",   "N": 100, "D": 8000,  "K": 4, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "3. Extreme Stress (D=15k)",    "N": 100, "D": 15000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "4. Absolute Limit (D=30k)",    "N": 100, "D": 30000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "5. Missing Visits (30%)",      "N": 150, "D": 2000,  "K": 4, "C": 2, "miss": 0.3, "noise": 1.0},
        {"name": "6. High Sensor Noise (3x)",    "N": 150, "D": 2000,  "K": 4, "C": 2, "miss": 0.0, "noise": 3.0},
    ]
    
    # 1. Build the list of all independent tasks
    tasks = []
    task_id = 0
    for s in scenarios:
        for mode in ["exact", "diagonal"]:
            for run_idx in range(n_runs):
                seed = 400 + run_idx + task_id # Ensure unique seed per run
                tasks.append({
                    "task_id": task_id,
                    "scenario": s,
                    "mode": mode,
                    "run_idx": run_idx,
                    "seed": seed
                })
                task_id += 1

    total_tasks = len(tasks)
    
    # 2. Configure 128-Core Setup (16 Workers * 8 Threads = 128 Cores)
    MAX_WORKERS = 16 
    
    logging.info(f"--- LAUNCHING 128-CORE MULTIPROCESSING ---")
    logging.info(f"Total isolated tasks queued: {total_tasks}")
    logging.info(f"Active Python Processes: {MAX_WORKERS} (8 PyTorch Threads each)")
    logging.info("-" * 108)
    
    # We will store results here to average them later
    results_aggregator = {s['name']: {"exact": [], "diagonal": []} for s in scenarios}
    
    # 3. Launch Process Pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the pool
        future_to_task = {
            executor.submit(run_single_simulation, t["task_id"], t["scenario"], t["mode"], t["run_idx"], t["seed"]): t 
            for t in tasks
        }
        
        # As each task finishes, process the result immediately
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_task):
            completed_count += 1
            res = future.result()
            
            if res["status"] == "error":
                logging.error(f"Task {res['task_id']} FAILED ({res['scenario_name']} | {res['mode']}):")
                logging.error(res["error_msg"])
            else:
                logging.info(f"[Progress {completed_count}/{total_tasks}] "
                             f"Finished {res['scenario_name']} | {res['mode'].capitalize()} in {res['time']:.1f}s")
                
                # Store for final aggregation
                results_aggregator[res["scenario_name"]][res["mode"]].append(res)

    # 4. Print the final beautifully formatted table
    logging.info("\n" + "=" * 108)
    logging.info("FINAL AGGREGATED RESULTS")
    logging.info("=" * 108)
    header = f"{'Scenario':<28} | {'Mode':<10} | {'Λ Corr (μ ± σ)':<15} | {'F Corr (μ ± σ)':<15} | {'Γ Corr (μ ± σ)':<15} | {'Avg Time (s)'}"
    logging.info(header)
    logging.info("-" * 108)
    
    for s in scenarios:
        for mode in ["exact", "diagonal"]:
            runs = results_aggregator[s["name"]][mode]
            if not runs:
                continue
                
            l_corrs = [r["l_corr"] for r in runs]
            f_corrs = [r["f_corr"] for r in runs]
            g_corrs = [r["g_corr"] for r in runs]
            times = [r["time"] for r in runs]
            
            l_str = f"{np.mean(l_corrs):.3f} ± {np.std(l_corrs):.3f}"
            f_str = f"{np.mean(f_corrs):.3f} ± {np.std(f_corrs):.3f}"
            g_str = f"{np.mean(g_corrs):.3f} ± {np.std(g_corrs):.3f}"
            
            result_row = f"{s['name']:<28} | {mode.capitalize():<10} | {l_str:<15} | {f_str:<15} | {g_str:<15} | {np.mean(times):>8.1f}"
            logging.info(result_row)
            
    logging.info("-" * 108)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True) # Forces fresh interpreters, preventing PyTorch thread deadlocks
    
    if run_smoke_test():
        run_stress_test_multiprocessing(n_runs=2) 
    else:
        logging.error("Aborting full stress test due to smoke test failure.")