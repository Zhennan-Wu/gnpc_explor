import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comp_utils import compute_sigma_diagonal


class OUDynamicFactorModel(nn.Module):
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M = D, K, M
        self.device = device
        
        # --- Observation Parameters ---
        self.Lambda = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        self.sigma_obs = nn.Parameter(torch.tensor(1.0, device=device)) 
        
        # --- OU Dynamics Parameters ---
        self.raw_rho = nn.Parameter(torch.ones(K, device=device) * 0.5)
        self.Gamma_raw = nn.Parameter(torch.eye(K, device=device) * 0.1)
        self.alpha = nn.Parameter(torch.zeros(K, device=device))            
        self.Phi = nn.Parameter(torch.zeros(K, M, device=device))           

        # --- Horseshoe Prior Parameters ---
        self.tau = torch.tensor(1.0, device=device)           
        self.v_dk = torch.ones(D, K, device=device)           
        self.c_reg = torch.tensor(1.0, device=device)

        self.history = {'mse': [], 'corr': []}
        self.to(device)

    def get_rho(self):
        # Added stability floor of 1e-4 (from dfa_ou2.py)
        return F.softplus(self.raw_rho) + 1e-4

    def get_gamma(self):
        L_offdiag = torch.tril(self.Gamma_raw, diagonal=-1)
        diag_elements = torch.diag(self.Gamma_raw) 
        # Added stability floor to diagonal (from dfa_ou2.py)
        L_diag = torch.diag(F.softplus(diag_elements) + 1e-4)
        return L_offdiag + L_diag

    def get_transition_params(self, delta_t, s_i, t_prev):
        dt = torch.clamp(delta_t, min=1e-6)
        rho = self.get_rho()
        gamma = self.get_gamma()
        
        A_ij = torch.exp(-rho * dt) 
        Sigma_ij = compute_sigma_diagonal(rho, gamma, dt)
        # Symmetrize and add small jitter for stability
        Sigma_ij = 0.5 * (Sigma_ij + Sigma_ij.T) + 1e-8 * torch.eye(self.K, device=self.device)
        
        # Time-varying drift
        mu_t = self.alpha + torch.mv(self.Phi, s_i) * t_prev
        mu_next = self.alpha + torch.mv(self.Phi, s_i) * (t_prev + dt)
        
        # OU mean integration
        G = (1 - A_ij) / (rho + 1e-9)
        b_ij = mu_next - A_ij * mu_t - G * torch.mv(self.Phi, s_i)
        
        return A_ij, b_ij, Sigma_ij

    def kalman_filter_smoother(self, x_i, times_i, s_i):
        Ji = x_i.size(0)
        f_filt = torch.zeros(Ji, self.K, device=self.device)
        P_filt = torch.zeros(Ji, self.K, self.K, device=self.device)
        f_smooth = torch.zeros(Ji, self.K, device=self.device)
        P_smooth = torch.zeros(Ji, self.K, self.K, device=self.device)
        P_lag1 = torch.zeros(Ji, self.K, self.K, device=self.device) 
        
        f_curr = torch.zeros(self.K, device=self.device) 
        P_curr = torch.eye(self.K, device=self.device) 
        f_filt[0], P_filt[0] = f_curr, P_curr

        # Clamp sigma_obs to prevent division by zero or tiny values
        sig_obs_stable = self.sigma_obs.clamp(min=1e-6)
        LTL = self.Lambda.T @ self.Lambda / sig_obs_stable

        # --- Forward Pass (Filtering) ---
        for j in range(1, Ji):
            dt = times_i[j] - times_i[j-1]
            A_vec, b, Q = self.get_transition_params(dt, s_i, times_i[j-1])
            
            # 1. Predict
            f_pred = A_vec * f_curr + b
            P_pred = A_vec[:, None] * P_curr * A_vec[None, :] + Q
            
            # 2. Woodbury Kalman Gain Update (Solve-based for stability)
            M = torch.eye(self.K, device=self.device) + P_pred @ LTL
            M = M + 1e-6 * torch.eye(self.K, device=self.device) 
            # LU-solve is more stable than Cholesky for ill-conditioned innovation
            K_gain = torch.linalg.solve(M, P_pred @ self.Lambda.T / sig_obs_stable)

            # 3. Update State and Covariance
            innovation = x_i[j] - self.Lambda @ f_pred
            f_curr = f_pred + K_gain @ innovation
            
            # Joseph form 
            IKL = torch.eye(self.K, device=self.device) - K_gain @ self.Lambda
            P_curr = IKL @ P_pred @ IKL.T + K_gain @ K_gain.T * sig_obs_stable
            
            # Forced Symmetrization (Crucial for stability)
            P_curr = 0.5 * (P_curr + P_curr.T)
            f_filt[j], P_filt[j] = f_curr, P_curr

        # --- Backward Pass ---
        f_smooth[-1], P_smooth[-1] = f_filt[-1], P_filt[-1]
        for j in range(Ji-2, -1, -1):
            dt = times_i[j+1] - times_i[j]
            A_next_v, b_next, Q_next = self.get_transition_params(dt, s_i, times_i[j])
            
            P_pred_next = A_next_v[:, None] * P_filt[j] * A_next_v[None, :] + Q_next
            # Add jitter before Cholesky in backward pass
            P_pred_next = 0.5 * (P_pred_next + P_pred_next.T) + 1e-7 * torch.eye(self.K, device=self.device)
            L_next = torch.linalg.cholesky(P_pred_next)
            
            rhs_smooth = A_next_v[:, None] * P_filt[j]
            J_T = torch.cholesky_solve(rhs_smooth, L_next)
            J = J_T.T 
            
            f_smooth[j] = f_filt[j] + J @ (f_smooth[j+1] - (A_next_v * f_filt[j] + b_next))
            P_smooth[j] = P_filt[j] + J @ (P_smooth[j+1] - P_pred_next) @ J.T
            P_smooth[j] = 0.5 * (P_smooth[j] + P_smooth[j].T)
            P_lag1[j+1] = J @ P_smooth[j+1] 
            
        return f_smooth, P_smooth, P_lag1

    def m_step(self, data, times, covs, all_f, all_P, all_P1):
        # 1. Batched Lambda Update
        with torch.no_grad():
            sum_ffT = torch.zeros(self.K, self.K, device=self.device)
            sum_xf = torch.zeros(self.D, self.K, device=self.device)
            
            for i in range(len(all_f)):
                outer_prods = torch.einsum('bik,bij->kj', all_f[i].unsqueeze(-1), all_f[i].unsqueeze(-2))
                sum_ffT += (outer_prods + all_P[i].sum(0))
                sum_xf += torch.einsum('ti,tj->ij', data[i], all_f[i])
            
            v_reg = (self.c_reg**2 * self.v_dk**2) / (self.c_reg**2 + self.tau**2 * self.v_dk**2 + 1e-9)
            prec_diag = 1.0 / (self.tau**2 * v_reg + 1e-9) 
            
            lhs = sum_ffT.unsqueeze(0) + torch.diag_embed(prec_diag) 
            lhs += 1e-6 * torch.eye(self.K, device=self.device)
            
            new_lambda = torch.linalg.solve(lhs, sum_xf.unsqueeze(-1)).squeeze(-1)
            self.Lambda.copy_(new_lambda)

            # Update Horseshoe Hyperpriors
            eta = 1.0 / (1.0 + self.v_dk**2 + 1e-9) 
            self.v_dk.copy_(torch.sqrt((1.0 + (self.Lambda**2/(self.tau**2 + 1e-9))) / (1.0 + eta)))
            xi = 1.0 / (1.0 + self.tau**2 + 1e-9) 
            self.tau.copy_(torch.sqrt((1.0 + (self.Lambda**2/(self.v_dk**2 + 1e-9)).sum()) / (self.D*self.K + 1.1 + xi)))

        # 2. Dynamics Update (Referencing dfa_ou2 logic for jitter in Q)
        dyn_optimizer = torch.optim.Adam([self.raw_rho, self.Gamma_raw, self.Phi, self.alpha], lr=5e-3)
        
        for _ in range(2):
            dyn_optimizer.zero_grad()
            loss = 0
            for i in range(len(all_f)):
                f_i, P_i, P1_i = all_f[i].detach(), all_P[i].detach(), all_P1[i].detach()
                dt_list = torch.clamp(times[i][1:] - times[i][:-1], min=1e-6)
                for j in range(1, f_i.size(0)):
                    A_v, b, Q = self.get_transition_params(dt_list[j-1], covs[i], times[i][j-1])
                    # Add jitter to Q for stable logdet/solve (similar to dfa_ou2)
                    L_Q = torch.linalg.cholesky(Q + 1e-7 * torch.eye(self.K, device=self.device))
                    
                    diff = f_i[j] - (A_v * f_i[j-1] + b)
                    A_mat = torch.diag(A_v)
                    err_cov = P_i[j] + A_mat @ P_i[j-1] @ A_mat.T - A_mat @ P1_i[j].T - P1_i[j] @ A_mat.T
                    
                    quad = torch.dot(diff, torch.cholesky_solve(diff.unsqueeze(1), L_Q).squeeze())
                    trace = torch.trace(torch.cholesky_solve(err_cov, L_Q))
                    logdet = 2.0 * L_Q.diagonal().log().sum()
                    
                    loss += logdet + quad + trace
                    loss += 1e-3 * torch.norm(self.Gamma_raw) 
                    loss += 1e-3 * torch.norm(1.0 / (self.get_rho() + 1e-6))
            loss.backward()
            dyn_optimizer.step()

            # 3. Observation Noise Update
            with torch.no_grad():
                rss, n_total = 0.0, 0
                LTL = self.Lambda.T @ self.Lambda
                for i in range(len(data)):
                    n_total += data[i].size(0)
                    err = data[i] - all_f[i] @ self.Lambda.T
                    rss += (err**2).sum() + torch.diagonal(all_P[i] @ LTL, dim1=-2, dim2=-1).sum()
                
                # Floor sig_obs to 1e-6 (from dfa_ou2)
                self.sigma_obs.copy_((rss / (self.D * n_total + 1e-9)).clamp(min=1e-6))

    def fit(self, data, covs, times, L_true, epochs=50, tol=1e-4):
        prev_L = None
        for epoch in range(epochs):
            all_f, all_P, all_P1 = [], [], []
            for i in range(len(data)):
                f, P, P1 = self.kalman_filter_smoother(data[i], times[i], covs[i])
                all_f.append(f); all_P.append(P); all_P1.append(P1)
            
            self.m_step(data, times, covs, all_f, all_P, all_P1)
            
            # Convergence check
            curr_L = self.Lambda.detach().clone()
            if prev_L is not None:
                delta = torch.norm(curr_L - prev_L) / torch.norm(prev_L)
                if delta < tol:
                    print(f"Converged at epoch {epoch}")
                    break
            prev_L = curr_L

            corr, mse = self.evaluate(L_true, data, times, covs)
            self.history['mse'].append(mse)
            self.history['corr'].append(corr)
            if epoch % 5 == 0:
                print(f"Epoch {epoch:02d} | MSE: {mse:.4f} | Loading Corr: {corr:.4f}")

    def evaluate(self, Lambda_true, data, times, covs):
        # Alignment via Procrustes for correlation check
        L_true_np = Lambda_true.cpu().numpy()
        L_est_np = self.Lambda.detach().cpu().numpy()
        U, _, Vt = scipy.linalg.svd(L_true_np.T @ L_est_np)
        L_aligned = torch.tensor(L_est_np @ (Vt.T @ U.T), device=self.device)
        
        corr = np.corrcoef(L_true_np.flatten(), L_aligned.cpu().numpy().flatten())[0, 1]
        rss, count = 0.0, 0
        for i in range(len(data)):
            f_s, _, _ = self.kalman_filter_smoother(data[i], times[i], covs[i])
            rss += torch.sum((data[i] - f_s @ L_aligned.T)**2)
            count += data[i].numel()
        return corr, (rss / count).item()
