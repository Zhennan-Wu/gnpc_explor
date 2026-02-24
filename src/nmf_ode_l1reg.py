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
        # Lambda: The factor loading matrix (D x K). Initialized with small noise.
        self._Lambda_unconstrained = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        
        # A: The transition matrix (K x K) defining the ODE dynamics: ds/dt = As + b
        self._A_unconstrained = nn.Parameter(torch.randn(K, K, device=device) * 0.1)
        
        # b: The bias/offset vector for latent transitions.
        self.b = nn.Parameter(torch.zeros(K, device=device))
        
        # Log-variance of the observation noise (for numerical stability).
        self.log_sigma_obs = nn.Parameter(torch.tensor(-2.0, device=device))
        
        # History tracker for metrics during training.
        self.history = {'mse': [], 'corr_lambda': [], 'err_A': [], 'err_sig': [], 'likelihood': []}
        self.to(device)

    def get_Lambda(self): 
        """
        Applies Softplus to ensure non-negativity and performs L2-normalization 
        on columns to prevent scale-drift between Lambda and latent states.
        """
        L = F.softplus(self._Lambda_unconstrained)
        return L / (torch.norm(L, dim=0, keepdim=True) + 1e-6)

    def get_A(self): 
        """
        Returns the transition matrix A. 
        Ensures stability by forcing the diagonal elements to be negative.
        """
        A = self._A_unconstrained
        # Subtracting softplus of the diagonal ensures negative eigenvalues (decaying systems).
        return A - torch.diag_embed(F.softplus(torch.diagonal(A)) + 1e-3)

    def get_history(self):
        """
        Returns the training history of metrics.
        """
        return self.history
    
    def propagate(self, s0, t_points):
        """
        Computes latent state trajectories using the Matrix Exponential solution 
        for a linear ODE: s(t+dt) = s(t) * exp(A*dt) + integration_constant.
        
        Args:
            s0: Initial latent state (1 x K).
            t_points: Time steps for the specific subject.
        """
        A, b = self.get_A(), self.b
        s_list, s_curr = [s0], s0
        
        for i in range(1, len(t_points)):
            # Calculate time delta between observations
            dt = torch.clamp(t_points[i] - t_points[i-1], min=1e-6)
            
            # Get discrete-time transition matrix (Fmat) and control input (u)
            Fmat, u = linear_ode_transition(A, b, dt)
            
            # State update: s_{t+1} = s_t @ F^T + u
            s_curr = (s_curr @ Fmat.T) + u
            s_list.append(s_curr)
            
        return torch.stack(s_list)

    def fit(self, data, covs, times, gt_params, epochs=100):
        """
        Optimizes model parameters to minimize reconstruction error and 
        regularization penalties.
        """
        num_subjects = len(data)
        # Latent initial states for each subject
        self.s0 = nn.Parameter(torch.randn(num_subjects, self.K, device=self.device) * 0.5)
        
        opt = torch.optim.Adam(self.parameters(), lr=1e-2)
        
        for epoch in range(epochs):
            opt.zero_grad()
            total_loss = 0
            
            for i in range(num_subjects):
                # 1. Propagate latent states through time
                s_traj = self.propagate(self.s0[i], times[i])
                
                # 2. Project latents to observation space: X = S * Lambda^T
                x_hat = s_traj @ self.get_Lambda().T
                
                # 3. Calculate Gaussian Log-Likelihood loss + L1 Sparsity on Lambda
                mse_term = ((data[i] - x_hat)**2).mean() / (2 * torch.exp(2 * self.log_sigma_obs))
                noise_reg = self.log_sigma_obs
                sparsity = 1e-4 * torch.norm(self.get_Lambda(), 1)
                
                total_loss += mse_term + noise_reg + sparsity
            
            total_loss.backward()
            opt.step()
            
            # Evaluation and Logging
            metrics = self.evaluate(gt_params, data, times, covs)
            metrics['likelihood'] = total_loss.item()
            for k in metrics: self.history[k].append(metrics[k])
            
            if epoch % 20 == 0:
                print(f"Ep {epoch:03d} | MSE: {metrics['mse']:.4f} | "
                      f"Loss: {total_loss.item():.2f} | L_Corr: {metrics['corr_lambda']:.4f}")

    def evaluate(self, gt, data, times, covs):
        """
        Evaluates model performance against Ground Truth (gt).
        Uses Procrustes Analysis (SVD-based rotation) to align latent spaces 
        since the model is invariant to rotation/permutation.
        """
        Le, Lt = self.get_Lambda().detach(), gt['Lambda']
        
        # Solve the Orthogonal Procrustes problem to align predicted Lambda with GT Lambda
        # R is the rotation matrix that minimizes ||Lt - Le @ R||
        U, _, Vt = scipy.linalg.svd(Lt.cpu().numpy().T @ Le.cpu().numpy())
        R = torch.tensor(Vt.T @ U.T, device=self.device, dtype=torch.float32)
        
        # Aligned Lambda
        La = Le @ R
        rss, count = 0, 0
        
        for i in range(len(data)):
            s = self.propagate(self.s0[i], times[i])
            # Calculate error in reconstructed space using the alignment matrix
            rss += (data[i] - (s @ R) @ La.T).pow(2).sum()
            count += data[i].numel()
            
        return {
            'mse': (rss/count).item(),
            'corr_lambda': np.corrcoef(Lt.cpu().numpy().flatten(), La.cpu().numpy().flatten())[0,1],
            'err_A': torch.norm(R.T @ self.get_A().detach() @ R - gt['A']).item(),
            'err_sig': torch.abs(torch.exp(self.log_sigma_obs) - gt['sigma_obs']).item()
        }