import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comp_utils import linear_ode_transition

class NMF_LinearODE_Model(nn.Module):
    def __init__(self, D, K, P, device='cpu'):
        super().__init__()
        self.D, self.K, self.P, self.device = D, K, P, device

        # Global Baseline (The "Simple Model" fallback)
        self.s0_global = nn.Parameter(torch.randn(K, device=device) * 0.1)

        # Shared Mean Encoder (Linear Model)
        self.init_mean_encoder = nn.Linear(P+1, K, device=device)
        
        # Learned Log-Variance (Initialized to a small value)
        self.log_var_init = nn.Parameter(torch.tensor([-2.0] * K, device=device))
        
        # ODE and NMF Parameters
        self._Lambda_unconstrained = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        self._A_unconstrained = nn.Parameter(torch.randn(K, K, device=device) * 0.1)
        self.b = nn.Parameter(torch.zeros(K, device=device))
        self.log_sigma_obs = nn.Parameter(torch.tensor(-2.0, device=device))
        
        self.history = {k: [] for k in ['mse', 'corr_lambda', 'err_theta', 'err_gamma', 'err_phi', 'err_alpha', 'err_sig', 'likelihood']}
        self.s0 = None 
        self.to(device)

    def get_Lambda(self): 
        L = F.softplus(self._Lambda_unconstrained)
        return L / (torch.norm(L, dim=0, keepdim=True) + 1e-6)

    def get_A(self): 
        A = self._A_unconstrained
        return A - torch.diag_embed(F.softplus(torch.diagonal(A)) + 1e-3)

    def get_history(self):
        return self.history
    
    def get_gamma(self):
        return self.log_var_init.exp()
    
    def get_phi(self):
        return self.init_mean_encoder.weight.data
    
    def get_alpha(self):
        return self.init_mean_encoder.bias.data
    
    def sample_s0(self, baseline_covs_tensor, init_time):
        """
        Computes mean from linear model and samples s0 using 
        the reparameterization trick.
        """

        mu_residual = self.init_mean_encoder(torch.cat([baseline_covs_tensor, init_time], dim=1))
        mu = self.s0_global + mu_residual
        std = torch.exp(0.5 * self.log_var_init)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def propagate(self, s0, t_points):
        s_curr = s0.unsqueeze(0) if s0.dim() == 1 else s0
        A, b = self.get_A(), self.b
        s_list = [s_curr]
        
        for i in range(1, len(t_points)):
            dt = torch.clamp(t_points[i] - t_points[i-1], min=1e-6)
            Fmat, u = linear_ode_transition(A, b, dt)
            s_curr = (s_curr @ Fmat.T) + u
            s_list.append(s_curr)
            
        return torch.stack(s_list).squeeze(1)

    def fit(self, data, baseline_covs, times, gt_params, epochs=100, lambda_reg=1e-4, l2_init=1e-3):
        opt = torch.optim.Adam(self.parameters(), lr=1e-2)
        
        # Stacking list of flat tensors into (M, P)
        baseline_covs_tensor = torch.stack(baseline_covs, dim=0).to(self.device)
        init_times = torch.tensor([times[i][0] for i in range(len(times))], device=self.device).reshape(-1, 1)
        
        for epoch in range(epochs):
            opt.zero_grad()
            
            # Stochastic initialization
            self.s0 = self.sample_s0(baseline_covs_tensor, init_times)
            
            total_loss = 0
            for i in range(len(data)):
                s_traj = self.propagate(self.s0[i], times[i])
                x_hat = s_traj @ self.get_Lambda().T
                
                mse_term = ((data[i] - x_hat)**2).mean() / (2 * torch.exp(2 * self.log_sigma_obs))
                noise_reg = self.log_sigma_obs
                sparsity = lambda_reg * torch.norm(self.get_Lambda(), 1)

                # L2 Regularization on the encoder weights only
                encoder_reg = l2_init * torch.norm(self.init_mean_encoder.weight, 2)
                total_loss += mse_term + noise_reg + sparsity + encoder_reg 
            
            total_loss.backward()
            opt.step()
            
            metrics = self.evaluate(gt_params, data, times, baseline_covs)
            metrics['likelihood'] = -total_loss.item()
            for k in metrics: self.history[k].append(metrics[k])
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:02d} | Loss: {total_loss.item():.2f} | MSE: {metrics['mse']:.4f} | Corr: {metrics['corr_lambda']:.4f} | Sig: {metrics['err_sig']:.4f}")

    def kalman_filter_smoother(self, obs, times, idx):
        """
        Modified to support visualizer which expects (T, K) output.
        Note: In this implementation, we use the first subject's initial state 
        as a representative for visualization if specific indices aren't provided.
        """
        # For visualization purposes, we use the learned initial state s0.
        # If this is called within the fit loop, self.s0[i] is passed directly.
        # If called by visualizer, we default to the first subject for the plot.
        s0_val = self.s0[idx] if hasattr(self, 's0') else torch.zeros(self.K, device=self.device)
        
        traj = self.propagate(s0_val, times)
        placeholder1 = None
        placeholder2 = None
        return traj, placeholder1, placeholder2 
    1
    def evaluate(self, gt, data, times, covs):
        Le, Lt = self.get_Lambda().detach(), gt['Lambda']
        U, _, Vt = scipy.linalg.svd(Lt.cpu().numpy().T @ Le.cpu().numpy())
        R = torch.tensor(Vt.T @ U.T, device=self.device, dtype=torch.float32)
        
        La = Le @ R
        rss, count = 0, 0
        s0_eval = self.s0.detach()
        
        for i in range(len(data)):
            s = self.propagate(s0_eval[i], times[i])
            reconstructed = (s @ R) @ La.T
            rss += (data[i] - reconstructed).pow(2).sum()
            count += data[i].numel()
            
        return {
            'mse': (rss/count).item(),
            'corr_lambda': np.corrcoef(Lt.cpu().numpy().flatten(), La.cpu().numpy().flatten())[0,1],
            'err_theta': torch.norm(R.T @ self.get_A().detach() @ R - gt['A']).item(),
            'err_sig': torch.abs(torch.exp(self.log_sigma_obs) - gt['sigma_obs']).item(),
            'err_gamma': torch.norm(self.get_gamma()-gt['Gamma']).item(),
            'err_phi': torch.norm(R.T @ self.get_phi().detach() - gt['Phi']).item(), 
            'err_alpha': torch.norm(R.T @ self.get_alpha().detach() - gt['alpha']).item()            
        }