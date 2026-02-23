import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from comp_utils import compute_sigma_diagonal


class OUDynamicFactorModel(nn.Module):
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M = D, K, M
        self.device = device
        
        # Observation params
        self.Lambda = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        self.log_sigma_obs = nn.Parameter(torch.log(torch.tensor(0.5, device=device))) 
        
        # Dynamics params
        self.raw_rho = nn.Parameter(torch.ones(K, device=device) * 0.5)
        self.Gamma_raw = nn.Parameter(torch.eye(K, device=device) * 0.1)
        self.alpha = nn.Parameter(torch.zeros(K, device=device))            
        self.Phi = nn.Parameter(torch.randn(K, M, device=device) * 0.05)           

        # Horseshoe Prior params
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))           
        self.log_v_dk = nn.Parameter(torch.zeros(D, K, device=device))           
        self.to(device)

    def get_rho(self): return F.softplus(self.raw_rho) + 1e-4
    
    def get_gamma(self):
        diag_part = torch.diag(F.softplus(torch.diag(self.Gamma_raw)) + 1e-4)
        return torch.tril(self.Gamma_raw, -1) + diag_part

    def get_transition_params(self, dt, s_i, t):
        rho = self.get_rho()
        A_v = torch.exp(-rho * dt)
        Q = compute_sigma_diagonal(rho, self.get_gamma(), dt)
        
        # Drift: mu_t = alpha + Phi*s*t
        drift_coeff = torch.mv(self.Phi, s_i)
        mu_t = self.alpha + drift_coeff * t
        mu_next = self.alpha + drift_coeff * (t + dt)
        b = mu_next - A_v * mu_t - ((1 - A_v)/rho) * drift_coeff
        return A_v, b, Q

    def kalman_filter_smoother(self, x, times, s):
            Ji = x.size(0)
            f_f = torch.zeros(Ji, self.K, device=self.device)
            P_f = torch.zeros(Ji, self.K, self.K, device=self.device)
            f_s = torch.zeros(Ji, self.K, device=self.device)
            P_s = torch.zeros(Ji, self.K, self.K, device=self.device)
            P_l = torch.zeros(Ji, self.K, self.K, device=self.device)
            
            f_curr, P_curr = torch.zeros(self.K, device=self.device), torch.eye(self.K, device=self.device)
            f_f[0], P_f[0] = f_curr, P_curr
            
            sig_sq = torch.exp(self.log_sigma_obs)
            inv_sig_sq = 1.0 / (sig_sq + 1e-9)
            LTL = self.Lambda.T @ self.Lambda  # K x K

            for j in range(1, Ji):
                A, b, Q = self.get_transition_params(times[j]-times[j-1], s, times[j-1])
                f_p = A * f_curr + b
                P_p = A[:, None] * P_curr * A[None, :] + Q # K x K
                
                # --- Efficient Kalman Gain (Woodbury-style) ---
                # We need (Lambda P_p Lambda.T + sigma^2 I)^-1 * (Lambda P_p)
                # Instead of DxD, we solve in KxK:
                # M = (sigma^2 * inv(P_p) + Lambda.T @ Lambda)
                # K_gain = P_p @ Lambda.T @ [ (1/sig^2) * (I - Lambda @ inv(M) @ Lambda.T @ (1/sig^2)) ]
                
                # More direct stable form for K_gain (K x D):
                # K_gain = inv(P_p^-1 + LTL/sig_sq) @ (Lambda.T / sig_sq)
                P_p_inv = torch.inverse(P_p + 1e-7 * torch.eye(self.K, device=self.device))
                M = P_p_inv + (LTL * inv_sig_sq)
                M_chol = torch.linalg.cholesky(M + 1e-7 * torch.eye(self.K, device=self.device))
                
                # Innovation (D)
                innov = x[j] - self.Lambda @ f_p 
                
                # Update Mean: f_curr = f_p + K_gain @ innov
                # K_gain @ innov = inv(M) @ (Lambda.T @ innov / sig_sq)
                rhs_m = (self.Lambda.T @ innov) * inv_sig_sq
                f_curr = f_p + torch.cholesky_solve(rhs_m.unsqueeze(1), M_chol).squeeze()
                
                # Update Covariance: P_curr = inv(M)
                P_curr = torch.cholesky_solve(torch.eye(self.K, device=self.device), M_chol)
                
                f_f[j], P_f[j] = f_curr, P_curr

            # --- Backward Pass remains KxK, so it is already efficient ---
            f_s[-1], P_s[-1] = f_f[-1], P_f[-1]
            for j in range(Ji-2, -1, -1):
                A, _, Q = self.get_transition_params(times[j+1]-times[j], s, times[j])
                P_p = A[:, None] * P_f[j] * A[None, :] + Q
                J = torch.linalg.solve(P_p, A[:, None] * P_f[j]).T
                f_s[j] = f_f[j] + J @ (f_s[j+1] - (A * f_f[j] + b))
                P_s[j] = P_f[j] + J @ (P_s[j+1] - P_p) @ J.T
                P_l[j+1] = J @ P_s[j+1]
                
            return f_s, P_s, P_l

    def m_step(self, data, times, covs, all_f, all_P, all_P1):
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
            
            for _ in range(5):
                optimizer.zero_grad()
                total_q = 0
                sig_sq = torch.exp(self.log_sigma_obs).detach()
                
                # Pre-calculate LTL to use the trace trick
                # This is K x K (e.g., 20x20), very small!
                LTL = self.Lambda.T @ self.Lambda 
                
                for i in range(len(data)):
                    f, P, P1 = all_f[i].detach(), all_P[i].detach(), all_P1[i].detach()
                    
                    # 1. Observation Term
                    # Part A: ||y - Lambda f||^2 (O(N*D*K) - No large matrices)
                    err = data[i] - f @ self.Lambda.T
                    res_sum_sq = err.pow(2).sum()
                    
                    # Part B: Trace trick (O(N*K^3) - Tiny memory footprint)
                    # Instead of Tr(Lambda @ P @ Lambda.T), we use Tr(LTL @ P)
                    # torch.sum(LTL.unsqueeze(0) * P) performs the trace efficiently
                    tr_obs = torch.sum(LTL.unsqueeze(0) * P)
                    
                    total_q += (res_sum_sq + tr_obs) / (2 * sig_sq)
                    
                    # 2. Dynamics Term (K x K operations only)
                    dt = times[i][1:] - times[i][:-1]
                    for j in range(1, f.size(0)):
                        A, b, Q = self.get_transition_params(dt[j-1], covs[i], times[i][j-1])
                        L_Q = torch.linalg.cholesky(Q + 1e-6 * torch.eye(self.K, device=self.device))
                        
                        diff = f[j] - (A * f[j-1] + b)
                        A_m = torch.diag(A)
                        e_cov = P[j] + A_m @ P[j-1] @ A_m.T - A_m @ P1[j].T - P1[j] @ A_m.T
                        
                        quad = torch.dot(diff, torch.cholesky_solve(diff.unsqueeze(1), L_Q).squeeze())
                        tr_dyn = torch.trace(torch.cholesky_solve(e_cov, L_Q))
                        total_q += 0.5 * (quad + tr_dyn + 2.0 * L_Q.diagonal().log().sum())

                # 3. Horseshoe Penalty (O(D*K))
                tau, v_dk = torch.exp(self.log_tau), torch.exp(self.log_v_dk)
                total_q += 0.5 * (self.Lambda.pow(2) / (tau**2 * v_dk**2 + 1e-9)).sum()
                total_q += torch.log(1 + v_dk**2).sum() + torch.log(1 + tau**2).sum()
                
                total_q.backward()
                optimizer.step()

            # Analytical sigma update (also using the trace trick)
            with torch.no_grad():
                rss, n_tot = 0, 0
                LTL = self.Lambda.T @ self.Lambda
                for i in range(len(data)):
                    n_tot += data[i].numel()
                    err = data[i] - all_f[i].detach() @ self.Lambda.T
                    tr = torch.sum(LTL.unsqueeze(0) * all_P[i].detach())
                    rss += err.pow(2).sum() + tr
                self.log_sigma_obs.copy_(torch.log(rss/n_tot + 1e-6))
                
            return total_q.item()

    def fit(self, data, covs, times, L_true, epochs=30):
        for epoch in range(epochs):
            all_f, all_P, all_P1 = [], [], []
            for i in range(len(data)):
                f, P, P1 = self.kalman_filter_smoother(data[i], times[i], covs[i])
                all_f.append(f); all_P.append(P); all_P1.append(P1)
            
            q = self.m_step(data, times, covs, all_f, all_P, all_P1)
            corr, mse = self.evaluate(L_true, data, times, covs)
            
            if epoch % 5 == 0:
                sig = torch.exp(self.log_sigma_obs).item()
                print(f"Epoch {epoch:02d} | Q: {q/1000:.2f}k | MSE: {mse:.4f} | Corr: {corr:.4f} | Sig: {sig:.4f}")

    def evaluate(self, L_t, data, times, covs):
        L_e = self.Lambda.detach().cpu().numpy()
        U, _, Vt = scipy.linalg.svd(L_t.cpu().numpy().T @ L_e)
        L_a = torch.tensor(L_e @ (Vt.T @ U.T), device=self.device)
        corr = np.corrcoef(L_t.cpu().numpy().flatten(), L_a.cpu().numpy().flatten())[0,1]
        rss, n = 0, 0
        for i in range(len(data)):
            f, _, _ = self.kalman_filter_smoother(data[i], times[i], covs[i])
            rss += (data[i] - f @ L_a.T).pow(2).sum()
            n += data[i].numel()
        return corr, (rss/n).item()

