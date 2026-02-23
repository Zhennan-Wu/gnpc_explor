import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_ode_transition(A, b, dt):
    """
    Exact discrete transition for linear ODE.

    ds/dt = A s + b
    """
    K = A.shape[0]
    I = torch.eye(K, device=A.device)

    Fmat = torch.matrix_exp(A * dt)

    # handle A^{-1}(F - I)b safely
    A_inv = torch.linalg.pinv(A)
    u = A_inv @ (Fmat - I) @ b

    return Fmat, u


def propagate_latent(A, b, s0, t_points):
    """
    s0: (N, K)
    returns: (N, T, K)
    """
    N, K = s0.shape
    T = len(t_points)

    s_list = [s0]
    s = s0

    for i in range(1, T):
        dt = t_points[i] - t_points[i - 1]
        Fmat, u = linear_ode_transition(A, b, dt)
        s = (s @ Fmat.T) + u
        s_list.append(s)

    return torch.stack(s_list, dim=1)


class NMF_LinearODE(nn.Module):
    def __init__(self, D, K):
        super().__init__()

        self.D = D
        self.K = K

        # -------------------------
        # NMF dictionary
        # -------------------------
        self._Lambda_unconstrained = nn.Parameter(
            torch.randn(D, K) * 0.1
        )

        # -------------------------
        # Linear ODE params
        # -------------------------
        self._A_unconstrained = nn.Parameter(
            torch.randn(K, K) * 0.1
        )
        self.b = nn.Parameter(torch.zeros(K))

        # initial latent
        self.s0 = nn.Parameter(torch.randn(1, K))

        # observation noise
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))

    # -------------------------
    # Nonnegative Lambda
    # -------------------------
    def Lambda(self):
        return F.softplus(self._Lambda_unconstrained)

    # -------------------------
    # Stable A (optional)
    # -------------------------
    def A(self):
        # encourage stability via negative diagonal shift
        A = self._A_unconstrained
        diag = torch.diag_embed(
            F.softplus(torch.diagonal(A)) + 1e-3
        )
        return A - diag

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, t_points, batch_size=1):
        A = self.A()
        Lambda = self.Lambda()

        s0 = self.s0.expand(batch_size, -1)

        s_traj = propagate_latent(A, self.b, s0, t_points)

        x_hat = torch.einsum("dk,ntk->ntd", Lambda, s_traj)

        return x_hat, s_traj

    # -------------------------
    # Loss
    # -------------------------
    def loss(self, x_obs, t_points):
        x_hat, _ = self.forward(t_points, x_obs.shape[0])

        sigma2 = torch.exp(2 * self.log_sigma)

        recon = ((x_obs - x_hat) ** 2).mean() / (2 * sigma2)
        reg = self.log_sigma

        return recon + reg
    

def train_model(model, x_obs, t_points, lr=1e-3, epochs=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        loss = model.loss(x_obs, t_points)
        loss.backward()
        opt.step()

        if ep % 100 == 0:
            print(f"Epoch {ep} | Loss {loss.item():.4f}")


if __name__ == "__main__":
    D = 20
    K = 3
    T = 50
    N = 5

    t_points = torch.linspace(0, 5, T)
    x_obs = torch.randn(N, T, D).abs()

    model = NMF_LinearODE(D, K)
    train_model(model, x_obs, t_points)

    x_hat, s_traj = model(t_points, batch_size=N)
    print(x_hat.shape)