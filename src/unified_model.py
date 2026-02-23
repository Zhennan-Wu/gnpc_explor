import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.linalg

class FullOUModel(nn.Module):
    def __init__(self, D, K, M, mode='DFA_OU'):
        super().__init__()
        self.D, self.K, self.M, self.mode = D, K, M, mode
        
        # 1. Observation Parameters
        self.Lambda = nn.Parameter(torch.randn(D, K) * 0.1)
        self.log_sigma_obs = nn.Parameter(torch.tensor(0.0))
        
        # 2. Dynamics: Full Matrix Theta (Decay) and shifts
        # Initializing Theta as identity to ensure stability (positive real eigenvalues)
        self.Theta = nn.Parameter(torch.eye(K) * 0.1) 
        self.alpha = nn.Parameter(torch.zeros(K))
        self.Phi = nn.Parameter(torch.zeros(K, M))
        
        # 3. Innovation: Cholesky factor L for Q = L @ L.T
        self.L_state = nn.Parameter(torch.eye(K) * 0.3) 

    def get_transition_params(self, delta_t, s_i, t_prev):
        """Exact matrix-based OU transitions."""
        dt = torch.clamp(delta_t, min=1e-6)
        
        # A = expm(-Theta * dt)
        A_ij = torch.matrix_exp(-self.Theta * dt)
        
        # Compute Sigma (Q) via Matrix Fraction Decomposition
        # This solves the Lyapunov integral for the transition covariance
        Q_inf = self.L_state @ self.L_state.T
        block_mat = torch.zeros(2 * self.K, 2 * self.K)
        block_mat[:self.K, :self.K] = self.Theta
        block_mat[:self.K, self.K:] = Q_inf
        block_mat[self.K:, self.K:] = -self.Theta.T
        
        expm_block = torch.matrix_exp(block_mat * dt)
        Sigma_ij = expm_block[:self.K, self.K:] @ torch.matrix_exp(self.Theta.T * dt)
        # Ensure symmetry and positive-definiteness
        Sigma_ij = (Sigma_ij + Sigma_ij.T) / 2.0 + 1e-6 * torch.eye(self.K)
        
        # Mean shift function
        mu_t = self.alpha + torch.mv(self.Phi, s_i) * t_prev
        mu_next = self.alpha + torch.mv(self.Phi, s_i) * (t_prev + dt)
        
        # b = mu_next - A @ mu_t - (I - A) @ (Theta^-1 @ Phi @ s_i)
        Theta_inv = torch.linalg.pinv(self.Theta)
        b_ij = mu_next - torch.mv(A_ij, mu_t) - torch.mv(Theta_inv @ (torch.eye(self.K) - A_ij) @ self.Phi, s_i)
        
        return A_ij, b_ij, Sigma_ij

    def kalman_filter_smoother(self, x_i, times_i, s_i):
        """Full covariance E-step."""
        Ji = x_i.size(0)
        f_filt, P_filt = torch.zeros(Ji, self.K), torch.zeros(Ji, self.K, self.K)
        f_smooth, P_smooth = torch.zeros(Ji, self.K), torch.zeros(Ji, self.K, self.K)
        P_lag1 = torch.zeros(Ji, self.K, self.K)
        
        f_curr, P_curr = torch.zeros(self.K), torch.eye(self.K)
        f_filt[0], P_filt[0] = f_curr, P_curr

        with torch.no_grad():
            for j in range(1, Ji):
                dt = times_i[j] - times_i[j-1]
                A, b, Q = self.get_transition_params(dt, s_i, times_i[j-1])
                
                f_pred = torch.mv(A, f_curr) + b
                P_pred = A @ P_curr @ A.T + Q
                
                # Gain calculation
                sigma_obs = torch.exp(self.log_sigma_obs)
                S = self.Lambda @ P_pred @ self.Lambda.T + sigma_obs * torch.eye(self.D)
                # K = P_pred @ Lambda.T @ S^-1
                K_gain = P_pred @ self.Lambda.T @ torch.inverse(S)
                
                f_curr = f_pred + torch.mv(K_gain, x_i[j] - torch.mv(self.Lambda, f_pred))
                P_curr = (torch.eye(self.K) - K_gain @ self.Lambda) @ P_pred
                f_filt[j], P_filt[j] = f_curr, P_curr

            # Backward Smoothing (RTS)
            f_smooth[-1], P_smooth[-1] = f_filt[-1], P_filt[-1]
            for j in range(Ji-2, -1, -1):
                dt = times_i[j+1] - times_i[j]
                A_next, _, Q_next = self.get_transition_params(dt, s_i, times_i[j])
                P_pred_next = A_next @ P_filt[j] @ A_next.T + Q_next
                
                J = P_filt[j] @ A_next.T @ torch.inverse(P_pred_next)
                f_smooth[j] = f_filt[j] + torch.mv(J, f_smooth[j+1] - (torch.mv(A_next, f_filt[j]) + b))
                P_smooth[j] = P_filt[j] + J @ (P_smooth[j+1] - P_pred_next) @ J.T
                P_lag1[j+1] = J @ P_smooth[j+1]
                
        return f_smooth, P_smooth, P_lag1

    def compute_q_function(self, data, times, covs, all_f, all_P, all_P1):
        """Expected Log-Likelihood objective for full covariance."""
        q_obs, q_state = 0, 0
        sigma_obs = torch.exp(self.log_sigma_obs)
        for i in range(len(data)):
            for j in range(data[i].size(0)):
                # Obs error: ||x - Lf||^2 + tr(L @ P @ L.T)
                err = data[i][j] - torch.mv(self.Lambda, all_f[i][j])
                q_obs += torch.sum(err**2) + torch.trace(self.Lambda @ all_P[i][j] @ self.Lambda.T)
                
                if j > 0:
                    dt = times[i][j] - times[i][j-1]
                    A, b, Q = self.get_transition_params(dt, covs[i], times[i][j-1])
                    Q_inv = torch.inverse(Q)
                    # Transition error
                    d_mean = all_f[i][j] - torch.mv(A, all_f[i][j-1]) - b
                    E_delta_delta = all_P[i][j] + A @ all_P[i][j-1] @ A.T - A @ all_P1[i][j].T - all_P1[i][j] @ A.T
                    
                    q_state += d_mean @ Q_inv @ d_mean + torch.trace(Q_inv @ E_delta_delta)
                    q_state += torch.logdet(Q)
                    
        penalty = 0.05 * torch.sum(torch.abs(self.Lambda))
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
    

def align_and_evaluate(L_true, L_est, data, times, covs, model):
    """Aligns factors via Procrustes rotation."""
    L_true_np = L_true.detach().numpy()
    L_est_np = L_est.detach().numpy()
    U, _, Vt = scipy.linalg.svd(L_true_np.T @ L_est_np)
    R = Vt.T @ U.T
    L_aligned = L_est_np @ R
    
    corr = np.corrcoef(L_true_np.flatten(), L_aligned.flatten())[0, 1]
    
    mse, n_total = 0, 0
    L_t = torch.tensor(L_aligned, dtype=torch.float32)
    for i in range(len(data)):
        f_s, _, _ = model.kalman_filter_smoother(data[i], times[i], covs[i])
        recon = f_s @ L_t.T
        mse += torch.sum((data[i] - recon)**2).item()
        n_total += data[i].numel()
    return corr, mse / n_total

def generate_full_test_data(N=10, D=30, K=3, M=2, mode='DFA_OU'):
    L_true = torch.randn(D, K)
    if mode == 'NMF_ODE': L_true = L_true.abs()
    L_true[torch.rand(D, K) < 0.6] = 0
    L_true /= (torch.norm(L_true, dim=0) + 1e-9)

    # Full Theta for cross-factor dependence
    Theta_true = torch.eye(K) * 0.1 + torch.randn(K, K) * 0.02
    Q_true = torch.eye(K) * 0.05 + 0.02 # Some covariance
    
    data, times, covs = [], [], []
    for i in range(N):
        Ji = 8
        dt = 0.5 + 2.0 * torch.rand(Ji)
        t = torch.cumsum(dt, dim=0)
        s_i = torch.randn(M)
        f_curr = torch.randn(K)
        f_list = []
        for j in range(Ji):
            step = dt[j] if j > 0 else 0.1
            A = torch.matrix_exp(-Theta_true * step)
            f_curr = torch.mv(A, f_curr) + torch.randn(K) * 0.1
            f_list.append(f_curr)
        data.append(torch.stack(f_list) @ L_true.T + torch.randn(Ji, D) * 0.1)
        times.append(t); covs.append(s_i)
    return data, times, covs, L_true


if __name__ == "__main__":
    D, K, M, N = 30, 3, 2, 10
    mode = 'DFA_OU'
    
    # 1. Generate Data
    data, times, covs, L_true = generate_full_test_data(N, D, K, M, mode=mode)
    
    # 2. Initialize Model
    model = FullOUModel(D, K, M, mode=mode)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # 3. Training
    print(f"Training Full Covariance OU-DFA...")
    for epoch in range(41):
        loss = model.train_step(data, times, covs, optimizer)
        if epoch % 10 == 0:
            corr, mse = align_and_evaluate(L_true, model.Lambda, data, times, covs, model)
            print(f"Epoch {epoch:02d} | Loss: {loss:.2f} | MSE: {mse:.4f} | Corr: {corr:.4f}")