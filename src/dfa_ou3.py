import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg

def compute_sigma_diagonal(rho, gamma, dt):
    """Computes transition covariance Q for OU: (1-exp(-2rt))/(2r) * Gamma*Gamma^T"""
    rho_stable = rho + 1e-9
    scale = (1 - torch.exp(-2 * rho_stable * dt)) / (2 * rho_stable)
    return (gamma @ gamma.T) * scale[:, None]

class BestOfBothOUDFM(nn.Module):
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M = D, K, M
        self.device = device
        
        # Observation: Log-space noise for better scaling in High-D
        self.Lambda = nn.Parameter(torch.randn(D, K, device=device) * 0.01)
        self.log_sigma_obs = nn.Parameter(torch.log(torch.tensor(0.5, device=device))) 
        
        # Dynamics: Floors for stability
        self.raw_rho = nn.Parameter(torch.ones(K, device=device) * 0.5)
        self.Gamma_raw = nn.Parameter(torch.eye(K, device=device) * 0.1)
        self.alpha = nn.Parameter(torch.zeros(K, device=device))            
        self.Phi = nn.Parameter(torch.zeros(K, M, device=device))           

        # Horseshoe Prior (Vectorized)
        self.tau = torch.tensor(1.0, device=device)           
        self.v_dk = torch.ones(D, K, device=device)           
        self.c_reg = torch.tensor(1.0, device=device)
        
        # Damping factor: the secret to signal retrieval in High-D
        self.eta_lambda = 0.4 
        self.to(device)

    def get_rho(self): 
        return F.softplus(self.raw_rho) + 1e-4
    
    def get_gamma(self):
        L_offdiag = torch.tril(self.Gamma_raw, diagonal=-1)
        L_diag = torch.diag(F.softplus(torch.diag(self.Gamma_raw)) + 1e-4)
        return L_offdiag + L_diag

    def get_transition_params(self, delta_t, s_i, t_p):
        dt = torch.clamp(delta_t, min=1e-6)
        rho, gamma = self.get_rho(), self.get_gamma()
        A = torch.exp(-rho * dt) 
        Q = compute_sigma_diagonal(rho, gamma, dt)
        Q = 0.5 * (Q + Q.T) + 1e-8 * torch.eye(self.K, device=self.device)
        
        drift = torch.mv(self.Phi, s_i)
        mu_t, mu_next = self.alpha + drift*t_p, self.alpha + drift*(t_p + dt)
        b = mu_next - A * mu_t - ((1 - A)/rho) * drift
        return A, b, Q

    def kalman_filter_smoother(self, x, times, s):
        T_len = x.size(0)
        f_f, P_f = torch.zeros(T_len, self.K, device=self.device), torch.zeros(T_len, self.K, self.K, device=self.device)
        f_s, P_s = torch.zeros(T_len, self.K, device=self.device), torch.zeros(T_len, self.K, self.K, device=self.device)
        P_l = torch.zeros(T_len, self.K, self.K, device=self.device) 
        
        f_c, P_c = torch.zeros(self.K, device=self.device), torch.eye(self.K, device=self.device) 
        f_f[0], P_f[0] = f_c, P_c
        sig = torch.exp(self.log_sigma_obs).clamp(min=1e-6)
        LTL = self.Lambda.T @ self.Lambda / sig

        # --- Forward (Joseph + LU) ---
        for j in range(1, T_len):
            A, b, Q = self.get_transition_params(times[j]-times[j-1], s, times[j-1])
            f_p = A * f_c + b
            P_p = A[:, None] * P_c * A[None, :] + Q
            
            # Robust LU Solve for Gain
            M_inv = torch.eye(self.K, device=self.device) + P_p @ LTL
            K_g = torch.linalg.solve(M_inv + 1e-7*torch.eye(self.K, device=self.device), P_p @ self.Lambda.T / sig)

            f_c = f_p + K_g @ (x[j] - self.Lambda @ f_p)
            IKL = torch.eye(self.K, device=self.device) - K_g @ self.Lambda
            P_c = 0.5 * (IKL @ P_p @ IKL.T + K_g @ K_g.T * sig + (IKL @ P_p @ IKL.T + K_g @ K_g.T * sig).T)
            f_f[j], P_f[j] = f_c, P_c

        # --- Backward (Standard Smoothing) ---
        f_s[-1], P_s[-1] = f_f[-1], P_f[-1]
        for j in range(T_len-2, -1, -1):
            A, b, Q = self.get_transition_params(times[j+1]-times[j], s, times[j])
            P_pn = A[:, None] * P_f[j] * A[None, :] + Q
            L = torch.linalg.cholesky(P_pn + 1e-7 * torch.eye(self.K, device=self.device))
            J = torch.cholesky_solve(A[:, None] * P_f[j], L).T
            f_s[j] = f_f[j] + J @ (f_s[j+1] - (A * f_f[j] + b))
            P_s[j] = 0.5 * (P_f[j] + J @ (P_s[j+1] - P_pn) @ J.T + (P_f[j] + J @ (P_s[j+1] - P_pn) @ J.T).T)
            P_l[j+1] = J @ P_s[j+1]
        return f_s, P_s, P_l

    def m_step(self, data, times, covs, all_f, all_P, all_P1):
        with torch.no_grad():
            sum_ffT, sum_xf = torch.zeros(self.K, self.K, device=self.device), torch.zeros(self.D, self.K, device=self.device)
            for i in range(len(all_f)):
                sum_ffT += (torch.einsum('ti,tj->ij', all_f[i], all_f[i]) + all_P[i].sum(0))
                sum_xf += (data[i].T @ all_f[i])
            
            # Horseshoe Prior Logic
            v_reg = (self.c_reg**2 * self.v_dk**2) / (self.c_reg**2 + self.tau**2 * self.v_dk**2 + 1e-9)
            prec = 1.0 / (self.tau**2 * v_reg + 1e-9)
            lhs = sum_ffT.unsqueeze(0) + torch.diag_embed(prec) + 1e-6*torch.eye(self.K, device=self.device)
            
            # Damped Update
            new_L = torch.linalg.solve(lhs, sum_xf.unsqueeze(-1)).squeeze(-1)
            self.Lambda.copy_((1 - self.eta_lambda) * self.Lambda + self.eta_lambda * new_L)

        # Dynamics Adam
        opt = torch.optim.Adam([self.raw_rho, self.Gamma_raw, self.Phi, self.alpha], lr=5e-3)
        for _ in range(3):
            opt.zero_grad()
            loss = 0
            for i in range(len(all_f)):
                f, P, P1, dt = all_f[i].detach(), all_P[i].detach(), all_P1[i].detach(), torch.clamp(times[i][1:]-times[i][:-1], 1e-6)
                for j in range(1, f.size(0)):
                    A, b, Q = self.get_transition_params(dt[j-1], covs[i], times[i][j-1])
                    LQ = torch.linalg.cholesky(Q)
                    df = f[j] - (A*f[j-1] + b)
                    Am = torch.diag(A)
                    ecov = P[j] + Am @ P[j-1] @ Am.T - Am @ P1[j].T - P1[j] @ Am.T
                    loss += 2.0*LQ.diagonal().log().sum() + torch.dot(df, torch.cholesky_solve(df.unsqueeze(1), LQ).squeeze()) + torch.trace(torch.cholesky_solve(ecov, LQ))
            loss.backward()
            opt.step()

        # Analytical Sigma Noise Update (Trace Trick)
        with torch.no_grad():
            rss, n_tot = 0, 0
            LTL = self.Lambda.T @ self.Lambda
            for i in range(len(data)):
                n_tot += data[i].numel()
                rss += (data[i] - all_f[i] @ self.Lambda.T).pow(2).sum() + torch.diagonal(all_P[i] @ LTL, dim1=-2, dim2=-1).sum()
            self.log_sigma_obs.copy_(torch.log((rss/n_tot).clamp(min=1e-6)))

    def fit(self, data, covs, times, L_true, epochs=30):
        for epoch in range(epochs):
            all_f, all_P, all_P1 = [], [], []
            for i in range(len(data)):
                f, P, P1 = self.kalman_filter_smoother(data[i], times[i], covs[i])
                all_f.append(f); all_P.append(P); all_P1.append(P1)
            self.m_step(data, times, covs, all_f, all_P, all_P1)
            
            if epoch % 5 == 0 or epoch == epochs-1:
                corr, mse = self.evaluate(L_true, data, times, covs)
                print(f"Epoch {epoch:02d} | Loading Corr: {corr:.4f} | MSE: {mse:.4f}")

    def evaluate(self, Lt, data, times, covs):
        Le = self.Lambda.detach().cpu().numpy()
        U, _, Vt = scipy.linalg.svd(Lt.cpu().numpy().T @ Le)
        La = torch.tensor(Le @ (Vt.T @ U.T), device=self.device)
        corr = np.corrcoef(Lt.cpu().numpy().flatten(), La.cpu().numpy().flatten())[0,1]
        rss, n = 0, 0
        for i in range(len(data)):
            f, _, _ = self.kalman_filter_smoother(data[i], times[i], covs[i])
            rss += (data[i] - f @ La.T).pow(2).sum()
            n += data[i].numel()
        return corr, (rss/n).item()

# --- High Dimensional Testing Script ---
def run_test():
    D, K, T, N = 8000, 10, 50, 2
    print(f"Generating Synthetic High-D Data: D={D}, K={K}")
    L_gt = torch.randn(D, K) * 0.1
    # ... (Generate factors via OU process and observations with noise) ...
    # (Simplified data gen for demo)
    data = [torch.randn(T, D) for _ in range(N)]
    times = [torch.linspace(0, 10, T) for _ in range(N)]
    covs = [torch.randn(2) for _ in range(N)]
    
    model = BestOfBothOUDFM(D, K, M=2)
    model.fit(data, covs, times, L_gt, epochs=30)

if __name__ == "__main__":
    run_test()