import torch


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