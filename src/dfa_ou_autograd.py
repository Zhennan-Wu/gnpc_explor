import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# --- Numerical Utilities ---

def safe_exprel_minus(x, eps=1e-8):
    """ Numerically stable (1 - exp(-x)) / x """
    return torch.where(x.abs() < eps, 1.0 - x/2.0 + (x**2)/6.0, (1.0 - torch.exp(-x)) / x)

def compute_sigma_diagonal(rho, gamma, delta_t):
    """ Integral of exp(-Theta r) Gamma Gamma^T exp(-Theta r) """
    Q_mat = torch.matmul(gamma, gamma.T)
    rho_sum = rho[:, None] + rho[None, :]
    return Q_mat * delta_t * safe_exprel_minus(rho_sum * delta_t)

# --- Model ---

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

# --- Data Generation ---

import torch
import numpy as np
import scipy.linalg

class StressTestGenerator:
    @staticmethod
    def generate(scenario="default", N=10, Ji=50, D=20, K=3, M=2, device='cpu'):
        """
        Scenarios:
        - 'high_noise': Low Signal-to-Noise ratio (Sigma_obs is large).
        - 'fast_dynamics': High rho (process reverts to mean quickly).
        - 'slow_dynamics': rho near zero (process behaves like a Random Walk).
        - 'sparse_loadings': Only a few observation dims respond to each factor.
        - 'massive_dims': Very high D relative to N and K.
        """
        torch.manual_seed(42)
        
        # --- Default Parameter Bounds ---
        obs_std = 0.1
        rho_range = (0.1, 0.5)
        gamma_scale = 0.1
        sparsity = 1.0 # 1.0 means dense
        
        if scenario == "high_noise":
            obs_std = 0.8  # Very difficult to recover factors
        elif scenario == "fast_dynamics":
            rho_range = (2.0, 5.0) # Factors oscillate/revert rapidly
        elif scenario == "slow_dynamics":
            rho_range = (0.001, 0.01) # Near Random Walk
        elif scenario == "sparse_loadings":
            sparsity = 0.2 # 80% of Lambda is zero
        elif scenario == "low_variance_dynamics":
            gamma_scale = 0.01 # Latent factors are almost deterministic
            
        # 1. Generate Loadings (Lambda)
        L_true = torch.randn(D, K, device=device)
        if sparsity < 1.0:
            mask = (torch.rand(D, K, device=device) < sparsity).float()
            L_true = L_true * mask
            
        # 2. Generate Dynamics Params
        Phi = torch.randn(K, M, device=device) * 0.05
        rho = (torch.rand(K, device=device) * (rho_range[1] - rho_range[0]) + rho_range[0])
        alpha = torch.randn(K, device=device) * 0.1
        
        data, times, covs = [], [], []
        
        for _ in range(N):
            t = torch.cumsum(torch.rand(Ji, device=device) * 0.5, dim=0)
            s = torch.randn(M, device=device)
            f = torch.zeros(Ji, K, device=device)
            
            # Gamma (Cholesky of state noise)
            Gamma = torch.eye(K, device=device) * gamma_scale
            
            for j in range(1, Ji):
                dt = t[j] - t[j-1]
                drift = alpha + Phi @ s * t[j]
                # OU Stochastic Differential Equation update
                noise = torch.randn(K, device=device) @ Gamma.T * torch.sqrt(dt)
                f[j] = f[j-1] + rho * (drift - f[j-1]) * dt + noise
                
            y = f @ L_true.T + torch.randn(Ji, D, device=device) * obs_std
            data.append(y); times.append(t); covs.append(s)
            
        return data, times, covs, L_true

# --- Test Execution Suite ---

def run_comprehensive_test():
    scenarios = ["high_noise", "slow_dynamics", "sparse_loadings", "fast_dynamics"]
    results = {}

    for sc in scenarios:
        print(f"\n--- Testing Scenario: {sc.upper()} ---")
        # Adjust dimensions for complexity
        D, K, M = (8000, 10, 3) if sc == "sparse_loadings" else (8000, 20, 2)
        
        data, times, covs, L_true = StressTestGenerator.generate(scenario=sc, D=D, K=K, M=M)
        model = OUDynamicFactorModel(D=D, K=K, M=M)
        
        # Train for fewer epochs for brevity in testing
        model.fit(data, covs, times, L_true, epochs=21)
        
        final_corr, final_mse = model.evaluate(L_true, data, times, covs)
        results[sc] = {"corr": final_corr, "mse": final_mse}

    print("\n" + "="*30)
    print("FINAL STRESS TEST SUMMARY")
    print("="*30)
    for sc, res in results.items():
        print(f"{sc:20}: Corr={res['corr']:.4f}, MSE={res['mse']:.4f}")

if __name__ == "__main__":
    run_comprehensive_test()