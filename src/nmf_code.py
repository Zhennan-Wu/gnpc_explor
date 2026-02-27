import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comp_utils import linear_ode_transition


class Conditional_NMF_ODE_Model(nn.Module):
    """
    Conditional Latent Linear ODE with NMF constraints and Low-Rank Covariate effects.
    """
    def __init__(self, D, K, num_covariates, device='cpu'):
        super().__init__()
        self.D, self.K, self.num_covariates, self.device = D, K, num_covariates, device

        # Global Baseline (The "Simple Model" fallback)
        self.s0_mean = nn.Parameter(torch.randn(K, device=device) * 0.1)
        self.log_var_init = nn.Parameter(torch.tensor([-2.0] * K, device=device))

        # --- NMF Spatial Factors (Fixed Basis) ---
        # Representing the 7 GM factors from the paper [cite: 115, 363]
        self._Lambda_unconstrained = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
    
        # --- Conditional ODE Parameters (Dynamics) ---
        # Base transition matrix (A_0)
        self._A_base = nn.Parameter(torch.randn(K, K, device=device) * 0.1)
        
        # Low-rank covariate components: M covariates map to (U, V) pairs
        # Each covariate j has two vectors u_j and v_j of size K
        self.U = nn.Parameter(torch.randn(num_covariates, K, device=device) * 0.05)
        self.V = nn.Parameter(torch.randn(num_covariates, K, device=device) * 0.05)
        
        self.b = nn.Parameter(torch.zeros(K, device=device))
        self.log_sigma_obs = nn.Parameter(torch.tensor(-2.0, device=device))
        
        # History tracker
        self.history = {k: [] for k in ['mse', 'corr_lambda', 'err_theta', 'err_gamma', 'err_phi', 'err_alpha', 'err_sig', 'err_b', 'likelihood']}
        self.to(device)

    def get_Lambda(self): 
        # NMF Non-negativity constraint [cite: 43, 431]
        L = F.softplus(self._Lambda_unconstrained)
        return L / (torch.norm(L, dim=0, keepdim=True) + 1e-6)

    def get_gamma(self):
        return self.log_var_init.exp()
    
    def get_phi(self):
        return self.U
    
    def get_alpha(self):
        return self.V

    def get_b(self):
        return self.b
    
    def get_conditional_A(self, cov_vector):
        """
        Constructs A = A_base + sum(cov_j * u_j @ v_j.T)
        Maps M covariates to KxK transition matrix via low-rank updates.
        """
        # A_base with negative diagonal for stability (decay)
        A = self._A_base - torch.diag_embed(F.softplus(torch.diagonal(self._A_base)) + 1e-3)
        
        # Apply low-rank covariate perturbations
        # cov_vector: (num_covariates,)
        for j in range(self.num_covariates):
            # Outer product of U[j] and V[j] scaled by covariate value
            perturbation = cov_vector[j] * torch.outer(self.U[j], self.V[j])
            A = A + perturbation
        return A

    def get_A(self):
        # Return the base A without covariate effects (for evaluation)
        A = self._A_base - torch.diag_embed(F.softplus(torch.diagonal(self._A_base)) + 1e-3)
        return A
    
    def get_history(self):
        return self.history

    def sample_s0(self, sample_size):
        """
        Computes mean from linear model and samples s0 using 
        the reparameterization trick.
        """
        mu = self.s0_mean
        std = torch.exp(0.5 * self.log_var_init)
        eps = torch.randn(sample_size, self.K, device=self.device)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)
    
    def propagate(self, s0, t_points, cov_vector):
        """
        Computes latent state trajectories conditioned on subject covariates.
        """
        s_curr = s0.view(1, self.K)
        A = self.get_conditional_A(cov_vector)
        b = self.b
        s_list = [s_curr]
        
        for i in range(1, len(t_points)):
            dt = torch.clamp(t_points[i] - t_points[i-1], min=1e-6)
            # Transition conditioned on subject-specific A
            Fmat, u = linear_ode_transition(A, b, dt)
            s_curr = (s_curr @ Fmat.T) + u
            s_list.append(s_curr)
            
        return torch.stack(s_list).squeeze(1)

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
        
        traj = self.propagate(s0_val, times, obs)
        placeholder1 = None
        placeholder2 = None
        return traj, placeholder1, placeholder2 
    
    def fit(self, data, covs, times, gt_params,  epochs=100, lambda_nmf=1e-4, lambda_cov=1e-3):
            num_subjects = len(data)
            opt = torch.optim.Adam(self.parameters(), lr=1e-2)
            
            # # Initial states (s0) representing baseline gray matter weights
            # self.s_init = nn.Parameter(torch.randn(num_subjects, self.K, device=self.device) * 0.1)

            for epoch in range(epochs):
                opt.zero_grad()
                total_reconstruction_loss = 0

                self.s0 = self.sample_s0(num_subjects) # Sample initial states for all subjects at once
                for i in range(num_subjects):
                    # 1. Propagate conditioned on subject covariates [cite: 509]
                    s_traj = self.propagate(self.s0[i], times[i], covs[i])
                    
                    # 2. Project back to voxel space using NMF Basis [cite: 489, 495]
                    x_hat = s_traj @ self.get_Lambda().T
                    
                    # 3. Gaussian Negative Log-Likelihood (MSE + Noise term) [cite: 514]
                    mse_term = ((data[i] - x_hat)**2).mean() / (2 * torch.exp(2 * self.log_sigma_obs))
                    total_reconstruction_loss += mse_term + self.log_sigma_obs

                # --- ADD REGULARIZATION HERE ---
                # Group Lasso on Covariates: Selects which traits drive the ODE dynamics
                cov_reg = 0
                for j in range(self.num_covariates):
                    cov_reg += torch.norm(self.U[j]) + torch.norm(self.V[j])
                
                # Sparsity on Lambda: Keeps brain factors spatially localized [cite: 429]
                nmf_reg = torch.norm(self.get_Lambda(), 1)
                
                # Combined Loss
                total_loss = total_reconstruction_loss + (lambda_cov * cov_reg) + (lambda_nmf * nmf_reg)
                
                # 4. Backprop and Step
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
        s0_eval = self.s0.detach()
        
        for i in range(len(data)):
            s = self.propagate(s0_eval[i], times[i], covs[i])
            reconstructed = (s @ R) @ La.T
            rss += (data[i] - reconstructed).pow(2).sum()
            count += data[i].numel()
            
        return {
            'mse': (rss/count).item(),
            'corr_lambda': np.corrcoef(Lt.cpu().numpy().flatten(), La.cpu().numpy().flatten())[0,1],
            'err_theta': torch.norm(R.T @ self.get_A().detach() @ R - gt['A']).item(),
            'err_sig': torch.abs(torch.exp(self.log_sigma_obs) - gt['sigma_obs']).item(),
            'err_gamma': torch.norm(self.get_gamma()-gt['Gamma']).item(),
            'err_phi': torch.norm(self.get_phi().detach()@R - gt['Phi']).item(), 
            'err_alpha': torch.norm(self.get_alpha().detach()@R - gt['alpha']).item(),
            'err_b': torch.norm(self.get_b() - gt['b']).item()
        }
    

import torch
import numpy as np
import matplotlib.pyplot as plt

def test_conditional_nmf_ode():
    # --- 1. Simulation Hyperparameters ---
    num_subjects = 10
    num_voxels = 100    # D (Observation space)
    num_factors = 3     # K (Latent space, e.g., Motor, Basal Ganglia, Cerebellum) [cite: 13]
    num_covariates = 2  # M (e.g., Age, Disease Duration) [cite: 425]
    num_timepoints = 5  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Running test on {device}...")

    # --- 2. Generate Synthetic Data ---
    # Random time points for each subject (longitudinal) [cite: 11]
    times = [torch.linspace(0, 2, num_timepoints, device=device) for _ in range(num_subjects)]
    
    # Random covariates (normalized) [cite: 526]
    covs = torch.randn(num_subjects, num_covariates, device=device)
    
    # Ground truth NMF Basis (Lambda) - Sparse and Non-negative [cite: 43, 431]
    gt_Lambda = torch.abs(torch.randn(num_voxels, num_factors, device=device))
    
    # Generate synthetic "voxel" observations (Data[i] = S_traj @ Lambda.T)
    data = []
    for i in range(num_subjects):
        # Create a simple decaying trajectory for testing
        s_init = torch.rand(num_factors, device=device)
        # Simulate a basic linear decay: s(t) = s0 * exp(-0.5 * t)
        s_traj = s_init.unsqueeze(0) * torch.exp(-0.5 * times[i].unsqueeze(1))
        voxels = s_traj @ gt_Lambda.T + torch.randn(num_timepoints, num_voxels, device=device) * 0.01
        data.append(voxels)

    # --- 3. Initialize Model ---
    model = Conditional_NMF_ODE_Model(
        D=num_voxels, 
        K=num_factors, 
        num_covariates=num_covariates, 
        device=device
    )

    # --- 4. Run Training (Fit) ---
    print("\nStarting Training Loop...")
    # Using the lambda values discussed for NMF and Covariate regularizations [cite: 429]
    model.fit(
        data=data, 
        covs=covs, 
        times=times, 
        epochs=50, 
        lambda_nmf=1e-3, 
        lambda_cov=1e-2
    )

    # --- 5. Verify and Visualize Results ---
    model.eval()
    with torch.no_grad():
        # Pick the first subject to check reconstruction
        test_subject_idx = 0
        s_pred = model.propagate(model.s_init[test_subject_idx], times[test_subject_idx], covs[test_subject_idx])
        x_reconstruction = s_pred @ model.get_Lambda().T
        
        mse = torch.mean((data[test_subject_idx] - x_reconstruction)**2).item()
        print(f"\nFinal Test Subject MSE: {mse:.6f}")

        # Check Sparsity of Lambda (NMF property) [cite: 43]
        lambda_matrix = model.get_Lambda()
        zero_threshold = 1e-3
        sparsity = (lambda_matrix < zero_threshold).float().mean().item()
        print(f"Lambda Matrix Sparsity: {sparsity*100:.1f}%")

        # Plotting the latent trajectories
        plt.figure(figsize=(10, 4))
        plt.plot(times[test_subject_idx].cpu(), s_pred.cpu())
        plt.title(f"Predicted Latent Factor Trajectories (Subject {test_subject_idx})")
        plt.xlabel("Years (Longitudinal)") 
        plt.ylabel("Factor Weights (GM Volume)") 
        plt.legend([f"Factor {i+1}" for i in range(num_factors)])
        plt.show()

if __name__ == "__main__":
    # Ensure the class definition from previous turns is in scope
    test_conditional_nmf_ode()