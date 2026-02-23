import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.linalg

# --- 1. THE MODEL ---
class AutogradLatentProcessModel(nn.Module):
    def __init__(self, D, K, M, mode='DFA_OU'):
        super().__init__()
        self.D, self.K, self.M, self.mode = D, K, M, mode
        
        # Parameters [cite: 31, 116, 118]
        self.Lambda = nn.Parameter(torch.randn(D, K) * 0.1)
        self.rho = nn.Parameter(torch.ones(K) * 0.05)
        self.alpha = nn.Parameter(torch.zeros(K))
        self.Phi = nn.Parameter(torch.zeros(K, M))
        
        # Variances in log-space for stability [cite: 835, 701]
        self.log_sigma_obs = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_state = nn.Parameter(torch.ones(K) * -2.3) # log(0.1)

    def get_transition_params(self, delta_t, s_i, t_prev):
        """[cite: 140, 181-183]"""
        dt = torch.clamp(delta_t, min=1e-6)
        A_ij = torch.exp(-self.rho * dt)
        sigma_state = torch.exp(self.log_sigma_state)
        Sigma_ij = torch.clamp((sigma_state / (2 * self.rho)) * (1 - torch.exp(-2 * self.rho * dt)), min=1e-8)
        
        mu_t = self.alpha + torch.mv(self.Phi, s_i) * t_prev
        mu_next = self.alpha + torch.mv(self.Phi, s_i) * (t_prev + dt)
        G = (1 - A_ij) / (self.rho + 1e-9)
        b_ij = mu_next - A_ij * mu_t - G * torch.mv(self.Phi, s_i)
        return A_ij, b_ij, Sigma_ij

    def kalman_filter_smoother(self, x_i, times_i, s_i):
        """[cite: 172-176, 184-205]"""
        Ji = x_i.size(0)
        f_filt, P_filt = torch.zeros(Ji, self.K), torch.zeros(Ji, self.K)
        f_smooth, P_smooth = torch.zeros(Ji, self.K), torch.zeros(Ji, self.K)
        P_lag1 = torch.zeros(Ji, self.K)
        
        f_curr, P_curr = torch.zeros(self.K), torch.ones(self.K) # [cite: 65]
        f_filt[0], P_filt[0] = f_curr, P_curr

        with torch.no_grad():
            for j in range(1, Ji):
                dt = times_i[j] - times_i[j-1]
                A, b, Q = self.get_transition_params(dt, s_i, times_i[j-1])
                f_pred = A * f_curr + b
                P_pred = (A**2) * P_curr + Q
                
                # Woodbury Identity [cite: 194]
                prec = 1.0 / (torch.exp(self.log_sigma_obs) + 1e-8)
                inner = torch.inverse(torch.diag(1.0/P_pred) + prec * (self.Lambda.T @ self.Lambda) + 1e-6*torch.eye(self.K))
                K_gain = prec * (inner @ self.Lambda.T)
                
                f_curr = f_pred + torch.mv(K_gain, x_i[j] - torch.mv(self.Lambda, f_pred))
                P_curr = P_pred - torch.diag(K_gain @ self.Lambda @ torch.diag(P_pred))
                f_filt[j], P_filt[j] = f_curr, P_curr

            f_smooth[-1], P_smooth[-1] = f_filt[-1], P_filt[-1]
            for j in range(Ji-2, -1, -1):
                dt = times_i[j+1] - times_i[j]
                A_next, b_next, Q_next = self.get_transition_params(dt, s_i, times_i[j])
                P_pred_next = (A_next**2) * P_filt[j] + Q_next
                J = (P_filt[j] * A_next) / (P_pred_next + 1e-9)
                f_smooth[j] = f_filt[j] + J * (f_smooth[j+1] - (A_next * f_filt[j] + b_next))
                P_smooth[j] = P_filt[j] + (J**2) * (P_smooth[j+1] - P_pred_next)
                P_lag1[j+1] = J * P_smooth[j+1]
        return f_smooth, P_smooth, P_lag1

    def compute_q_function(self, data, times, covs, all_f, all_P, all_P1):
        """[cite: 742-751]"""
        q_obs, q_state = 0, 0
        sigma_obs = torch.exp(self.log_sigma_obs)
        for i in range(len(data)):
            for j in range(data[i].size(0)):
                # Observation Term [cite: 765]
                err = data[i][j] - torch.mv(self.Lambda, all_f[i][j])
                q_obs += torch.sum(err**2) + torch.sum((self.Lambda**2) @ all_P[i][j])
                if j > 0:
                    # State Transition Term [cite: 771-778]
                    dt = times[i][j] - times[i][j-1]
                    A, b, Q = self.get_transition_params(dt, covs[i], times[i][j-1])
                    d_mean = all_f[i][j] - A * all_f[i][j-1] - b
                    d_cov = all_P[i][j] + (A**2) * all_P[i][j-1] - 2 * A * all_P1[i][j]
                    q_state += torch.sum((d_mean**2) / Q) + torch.sum(d_cov / Q) + torch.sum(torch.log(Q))
        
        # Sparsity penalty (Horseshoe proxy) [cite: 72]
        penalty = 0.01 * torch.sum(torch.abs(self.Lambda))
        return (q_obs / sigma_obs) + (self.D * self.log_sigma_obs) + q_state + penalty

    def train_step(self, data, times, covs, optimizer):
        all_f, all_P, all_P1 = [], [], []
        for i in range(len(data)):
            f, P, P1 = self.kalman_filter_smoother(data[i], times[i], covs[i])
            all_f.append(f); all_P.append(P); all_P1.append(P1)
        optimizer.zero_grad()
        loss = self.compute_q_function(data, times, covs, all_f, all_P, all_P1)
        loss.backward()
        optimizer.step()
        if self.mode == 'NMF_ODE':
            with torch.no_grad(): self.Lambda.clamp_(min=0.0)
        return loss.item()

# --- 2. THE UTILITIES ---

def align_and_evaluate(Lambda_true, Lambda_est, data, times, covs, model):
    """Aligns factors using Orthogonal Procrustes and calculates metrics [cite: 315-321]."""
    L_true_np = Lambda_true.detach().cpu().numpy()
    L_est_np = Lambda_est.detach().cpu().numpy()
    
    # Orthogonal Procrustes [cite: 315]
    U, _, Vt = scipy.linalg.svd(L_true_np.T @ L_est_np)
    R = Vt.T @ U.T
    L_aligned = L_est_np @ R
    
    # Pearson Correlation [cite: 316]
    corr = np.corrcoef(L_true_np.flatten(), L_aligned.flatten())[0, 1]
    
    # Reconstruction MSE [cite: 321]
    rss, n_total = 0, 0
    L_aligned_t = torch.tensor(L_aligned, dtype=torch.float32)
    for i in range(len(data)):
        f_s, _, _ = model.kalman_filter_smoother(data[i], times[i], covs[i])
        for j in range(data[i].size(0)):
            recon = torch.mv(L_aligned_t, f_s[j])
            rss += torch.sum((data[i][j] - recon)**2).item()
            n_total += data[i].size(1)
    return corr, rss / n_total

def generate_test_data(N=10, D=20, K=3, M=2, mode='DFA_OU'):
    """[cite: 243-253, 288-295]"""
    if mode == 'NMF_ODE': L_true = torch.abs(torch.randn(D, K))
    else: L_true = torch.randn(D, K)
    L_true[torch.rand(D, K) < 0.7] = 0.0 # [cite: 289]
    L_true /= (torch.norm(L_true, dim=0) + 1e-9)

    data, times, covs = [], [], []
    rho_true = 0.05 + 0.03 * torch.randn(K).abs() # [cite: 267]
    for i in range(N):
        Ji = 8
        dt = 0.5 + (2.0 * torch.rand(Ji))
        t = torch.cumsum(dt, dim=0)
        s_i = torch.randn(M)
        f = torch.zeros(Ji, K)
        f_curr = torch.randn(K)
        for j in range(Ji):
            step = dt[j] if j > 0 else 0.1
            A = torch.exp(-rho_true * step)
            Q = (0.1 / (2 * rho_true)) * (1 - torch.exp(-2 * rho_true * step))
            f_curr = A * f_curr + torch.randn(K) * torch.sqrt(Q)
            f[j] = f_curr
        data.append(f @ L_true.T + torch.randn(Ji, D) * 0.1)
        times.append(t); covs.append(s_i)
    return data, times, covs, L_true

# --- 3. THE TEST RUNNER ---

if __name__ == "__main__":
    D, K, M, N = 20, 3, 2, 10
    for mode in ['DFA_OU', 'NMF_ODE']:
        print(f"\n--- Testing Mode: {mode} ---")
        data, times, covs, L_true = generate_test_data(N, D, K, M, mode=mode)
        model = AutogradLatentProcessModel(D, K, M, mode=mode)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        
        for epoch in range(31):
            loss = model.train_step(data, times, covs, optimizer)
            if epoch % 10 == 0:
                corr, mse = align_and_evaluate(L_true, model.Lambda, data, times, covs, model)
                print(f"Epoch {epoch:02d} | Loss: {loss:.2f} | MSE: {mse:.4f} | Corr: {corr:.4f}")