import torch
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def safe_exprel_minus(x, eps=1e-8):
    """ Numerically stable (1 - exp(-x)) / x """
    return torch.where(
        x.abs() < eps,
        1.0 - x/2.0 + (x**2)/6.0,
        (1.0 - torch.exp(-x)) / x
    )

def compute_sigma_diagonal(rho, gamma, delta_t):
    """
    Computes the OU covariance Sigma_ij = integral exp(-Theta r) Gamma Gamma^T exp(-Theta r) dr
    where Theta = diag(rho).
    """
    Q = torch.matmul(gamma, gamma.T)
    # Pairwise sum of rho: (rho_i + rho_j)
    rho_sum = rho[:, None] + rho[None, :]
    x = rho_sum * delta_t
    
    stability_factor = safe_exprel_minus(x)
    sigma = Q * delta_t * stability_factor
    return sigma


def linear_ode_transition(A, b, dt):
    """Exact discrete transition for linear ODE: ds/dt = A s + b"""
    K = A.shape[0]
    I = torch.eye(K, device=A.device)
    # Using matrix exponential for the exact solution
    Fmat = torch.matrix_exp(A * dt)
    # Pseudoinverse handles potential singularities in A
    A_inv = torch.linalg.pinv(A)
    u = A_inv @ (Fmat - I) @ b
    return Fmat, u

def bridge_to_jax(all_data, all_times):
    lengths = [d.shape[0] for d in all_data]
    N_total = sum(lengths)
    y_jax = jnp.array(torch.cat(all_data, dim=0).cpu().numpy())
    t_jax = jnp.array(torch.cat(all_times, dim=0).cpu().numpy())
    is_start = np.zeros(N_total, dtype=bool)
    curr = 0
    for l in lengths:
        is_start[curr] = True
        curr += l
    return {'N_total': int(N_total), 't': t_jax, 'y': y_jax, 'is_start': jnp.array(is_start)}, lengths

class NumPyroModelWrapper:
    def __init__(self, samples, lengths, gt_params=None, y_true=None):
        self.samples = samples
        self.lengths = lengths
        self.gt = gt_params
        self.y_true = y_true 
        self._subj_ptr = 0
        self.device = "cpu"
        
        # Collapse sample dimension immediately to prevent shape errors
        self.x_mean_all = np.array(jnp.mean(samples['x'], axis=0))
        self.lambda_mean = np.array(jnp.mean(samples['Lambda'], axis=0))
        self.theta_mean = np.array(jnp.mean(samples['theta'], axis=0))
        self.sig_obs_mean = float(jnp.mean(samples['sigma_obs']))

    def get_Lambda(self):
        return torch.from_numpy(self.lambda_mean).float()

    def get_history(self):
        # Return the full metric set expected by plot_multi_model_metrics
        y_gt = self.y_true.cpu().numpy() if torch.is_tensor(self.y_true) else self.y_true
        y_pred = self.x_mean_all @ self.lambda_mean.T
        mse = np.mean((y_gt - y_pred)**2)
        
        # Procrustes Alignment for correlation metric
        L_gt = self.gt['Lambda'].cpu().numpy()
        U, _, Vt = scipy.linalg.svd(L_gt.T @ self.lambda_mean)
        R = Vt.T @ U.T
        corr_l = np.corrcoef(L_gt.flatten(), (self.lambda_mean @ R).flatten())[0, 1]

        return {
            'mse': [mse], 'corr_lambda': [corr_l], 
            'err_theta': [np.mean((self.theta_mean - self.gt['rho'].cpu().numpy())**2)],
            'err_gamma': [0.0], 'err_phi': [0.0], 'err_alpha': [0.0],
            'err_sig': [(self.sig_obs_mean - float(self.gt['sigma_obs']))**2],
            'likelihood': [0.0]
        }

    def kalman_filter_smoother(self, data, times, covs):
        # Returns (T, K) posterior mean. visual.py slices this into (T,) for plotting.
        start = sum(self.lengths[:self._subj_ptr])
        end = start + self.lengths[self._subj_ptr]
        x_slice = self.x_mean_all[start:end]
        self._subj_ptr = (self._subj_ptr + 1) % len(self.lengths)
        return torch.from_numpy(x_slice).float(), None, None

    def eval(self): pass