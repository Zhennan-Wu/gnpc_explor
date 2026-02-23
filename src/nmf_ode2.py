import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comp_utils import linear_ode_transition


class NMF_LinearODE_Model(nn.Module):
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M = D, K, M
        self.device = device

        # NMF Loadings: Initialization
        self._Lambda_unconstrained = nn.Parameter(torch.randn(D, K, device=device) * 0.1)

        # Stable ODE Parameters
        self._A_unconstrained = nn.Parameter(torch.randn(K, K, device=device) * 0.1)
        self.b = nn.Parameter(torch.zeros(K, device=device))
        
        # Observation noise
        self.log_sigma_obs = nn.Parameter(torch.tensor(-2.0, device=device))
        
        # Subject-specific initial states
        self.s0 = None 

        self.to(device)

    def get_Lambda(self):
        # Adjustment 1: Softplus + Column Normalization to fix scale drift
        L = F.softplus(self._Lambda_unconstrained)
        return L / (torch.norm(L, dim=0, keepdim=True) + 1e-6)

    def get_A(self):
        # Adjustment 2: Enforce stability via negative diagonal shift
        A = self._A_unconstrained
        diag = torch.diag_embed(F.softplus(torch.diagonal(A)) + 1e-3)
        return A - diag

    def propagate(self, s0, t_points):
        """Generates the full latent trajectory using exact integration"""
        A = self.get_A()
        s_list = [s0]
        s_curr = s0
        for i in range(1, len(t_points)):
            dt = torch.clamp(t_points[i] - t_points[i-1], min=1e-6)
            Fmat, u = linear_ode_transition(A, self.b, dt)
            s_curr = (s_curr @ Fmat.T) + u
            s_list.append(s_curr)
        return torch.stack(s_list)

    def fit(self, data, covs, times, L_true, epochs=300):
        num_subjects = len(data)
        # Parameterize subject-specific starting points
        self.s0 = nn.Parameter(torch.randn(num_subjects, self.K, device=self.device) * 0.5)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # Adjustment 3: LR Scheduler to stabilize late-stage training
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
        
        print(f"Starting Robust NMF-LinearODE Training (Target Epochs: {epochs})...")

        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0
            epoch_mse = 0
            
            for i in range(num_subjects):
                s_traj = self.propagate(self.s0[i], times[i])
                x_hat = s_traj @ self.get_Lambda().T
                
                # Reconstruction Loss
                sigma2 = torch.exp(2 * self.log_sigma_obs)
                recon = ((data[i] - x_hat) ** 2).mean() / (2 * sigma2)
                
                # Adjustment 4: L1 Sparsity Penalty (mimics Horseshoe effect)
                l1_penalty = 1e-4 * torch.norm(self.get_Lambda(), 1)
                
                total_loss += (recon + self.log_sigma_obs + l1_penalty)
                epoch_mse += F.mse_loss(x_hat, data[i]).item()
            
            total_loss.backward()
            optimizer.step()
            
            avg_mse = epoch_mse / num_subjects
            scheduler.step(avg_mse)

            if epoch % 20 == 0 or epoch == epochs - 1:
                corr, _ = self.evaluate(L_true, data, times, covs)
                print(f"Epoch {epoch:03d} | MSE: {avg_mse:.4f} | Loading Corr: {corr:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    def evaluate(self, Lt, data, times, covs):
        Le = self.get_Lambda().detach().cpu().numpy()
        Lt_np = Lt.cpu().numpy()
        
        # Align Estimated Lambda to True Lambda using Orthogonal Procrustes
        U, _, Vt = scipy.linalg.svd(Lt_np.T @ Le)
        La = torch.tensor(Le @ (Vt.T @ U.T), device=self.device)
        
        corr = np.corrcoef(Lt_np.flatten(), La.cpu().numpy().flatten())[0,1]
        
        rss, n = 0, 0
        with torch.no_grad():
            for i in range(len(data)):
                s_traj = self.propagate(self.s0[i], times[i])
                rss += (data[i] - s_traj @ La.T).pow(2).sum()
                n += data[i].numel()
        return corr, (rss/n).item()