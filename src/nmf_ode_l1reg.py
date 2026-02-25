import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comp_utils import linear_ode_transition


class NMF_LinearODE_Model(nn.Module):
    """
    A Latent Linear ODE model with NMF-inspired constraints.
    
    Attributes:
        D (int): Observation dimensionality (number of features).
        K (int): Latent space dimensionality (number of components).
        M (int): Number of subjects/samples.
    """
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M, self.device = D, K, M, device
        
        # --- Trainable Parameters ---
        self._Lambda_unconstrained = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        self._A_unconstrained = nn.Parameter(torch.randn(K, K, device=device) * 0.1)
        self.b = nn.Parameter(torch.zeros(K, device=device))
        self.log_sigma_obs = nn.Parameter(torch.tensor(-2.0, device=device))
        
        # History tracker
        self.history = {'mse': [], 'corr_lambda': [], 'err_theta': [], 'err_sig': [], 'likelihood': []}
        self.to(device)

    def get_Lambda(self): 
        L = F.softplus(self._Lambda_unconstrained)
        return L / (torch.norm(L, dim=0, keepdim=True) + 1e-6)

    def get_A(self): 
        A = self._A_unconstrained
        return A - torch.diag_embed(F.softplus(torch.diagonal(A)) + 1e-3)

    def get_history(self):
        return self.history
    
    def propagate(self, s0, t_points):
        """
        Computes latent state trajectories.
        Corrected to return shape (T, K) instead of (T, 1, K).
        """
        # Ensure s0 is (1, K) for consistent matrix multiplication
        if s0.dim() == 1:
            s_curr = s0.unsqueeze(0)
        else:
            s_curr = s0

        A, b = self.get_A(), self.b
        s_list = [s_curr]
        
        for i in range(1, len(t_points)):
            dt = torch.clamp(t_points[i] - t_points[i-1], min=1e-6)
            # Fmat: (K, K), u: (1, K)
            Fmat, u = linear_ode_transition(A, b, dt)
            
            # State update: (1, K) @ (K, K) + (1, K)
            s_curr = (s_curr @ Fmat.T) + u
            s_list.append(s_curr)
            
        # Stacked: (T, 1, K) -> Squeeze to (T, K)
        return torch.stack(s_list).squeeze(1)

    def kalman_filter_smoother(self, obs, times, covs):
        """
        Modified to support visualizer which expects (T, K) output.
        Note: In this implementation, we use the first subject's initial state 
        as a representative for visualization if specific indices aren't provided.
        """
        # For visualization purposes, we use the learned initial state s0.
        # If this is called within the fit loop, self.s0[i] is passed directly.
        # If called by visualizer, we default to the first subject for the plot.
        s0_val = self.s0[0] if hasattr(self, 's0') else torch.zeros(self.K, device=self.device)
        
        traj = self.propagate(s0_val, times)
        placeholder1 = None
        placeholder2 = None
        return traj, placeholder1, placeholder2 
    
    def fit(self, data, covs, times, gt_params, epochs=100):
        num_subjects = len(data)
        self.s0 = nn.Parameter(torch.randn(num_subjects, self.K, device=self.device) * 0.5)
        
        opt = torch.optim.Adam(self.parameters(), lr=1e-2)
        
        for epoch in range(epochs):
            opt.zero_grad()
            total_loss = 0
            
            for i in range(num_subjects):
                # 1. Propagate: returns (T, K)
                s_traj = self.propagate(self.s0[i], times[i])
                
                # 2. Project: (T, K) @ (K, D) -> (T, D)
                x_hat = s_traj @ self.get_Lambda().T
                
                # 3. Calculate Loss: data[i] is (T, D)
                mse_term = ((data[i] - x_hat)**2).mean() / (2 * torch.exp(2 * self.log_sigma_obs))
                noise_reg = self.log_sigma_obs
                sparsity = 1e-4 * torch.norm(self.get_Lambda(), 1)
                
                total_loss += mse_term + noise_reg + sparsity
            
            total_loss.backward()
            opt.step()
            
            # Evaluation and Logging
            metrics = self.evaluate(gt_params, data, times, covs)
            metrics['likelihood'] = -total_loss.item()
            for k in metrics: self.history[k].append(metrics[k])
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:02d} | Loss: {total_loss.item():.2f} | MSE: {metrics['mse']:.4f} | Corr: {metrics['corr_lambda']:.4f} | Sig: {metrics['err_sig']:.4f}")

    def evaluate(self, gt, data, times, covs):
        Le, Lt = self.get_Lambda().detach(), gt['Lambda']
        
        U, _, Vt = scipy.linalg.svd(Lt.cpu().numpy().T @ Le.cpu().numpy())
        R = torch.tensor(Vt.T @ U.T, device=self.device, dtype=torch.float32)
        
        La = Le @ R
        rss, count = 0, 0
        
        for i in range(len(data)):
            # s is (T, K)
            s = self.propagate(self.s0[i], times[i])
            # (T, K) @ (K, K) @ (K, D) -> (T, D)
            reconstructed = (s @ R) @ La.T
            rss += (data[i] - reconstructed).pow(2).sum()
            count += data[i].numel()
            
        return {
            'mse': (rss/count).item(),
            'corr_lambda': np.corrcoef(Lt.cpu().numpy().flatten(), La.cpu().numpy().flatten())[0,1],
            'err_theta': torch.norm(R.T @ self.get_A().detach() @ R - gt['A']).item(),
            'err_sig': torch.abs(torch.exp(self.log_sigma_obs) - gt['sigma_obs']).item()
        }