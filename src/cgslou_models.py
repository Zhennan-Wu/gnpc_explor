import jax
import jax.numpy as jnp
from jax import lax
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpyro.infer import log_likelihood


class LOU:
    def __init__(self, K, D, target_accept_prob=0.95):
        self.K = K
        self.D = D
        self.target_accept_prob = target_accept_prob
        self.mcmc = None
        
    def model(self, Nsub, N_total, start_idx, end_idx, t, y):
        # Basic OU Priors (Constant Reversion Target mu0)
        theta = numpyro.sample("theta", dist.LogNormal(jnp.log(1.0), 0.3).expand([self.K]))
        mu0 = numpyro.sample("mu0", dist.Normal(0.0, 2.0).expand([self.K]))
        sigma0 = numpyro.sample("sigma0", dist.HalfNormal(0.5).expand([self.K]))
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.2))

        # Standard Normal Prior for Lambda (Non-Sparse)
        Lambda = numpyro.sample("Lambda", dist.Normal(0.0, 0.5).expand([self.D, self.K]))

        x_all = []
        steps_per_sub = N_total // Nsub 

        for i in range(Nsub):
            # FIXED: Added scope to ensure unique site names per subject
            with numpyro.handlers.scope(prefix=f"sub_{i}"):
                s_idx = start_idx[i]
                # FIXED: Using dynamic_slice for JAX JIT compatibility
                subject_t = lax.dynamic_slice(t, (s_idx,), (steps_per_sub,))
                
                x0 = numpyro.sample("x_init", dist.Normal(mu0, sigma0))
                
                def transition_fn(carry, inputs):
                    x_prev = carry
                    t_prev, t_curr = inputs
                    dt = t_curr - t_prev
                    phi = jnp.exp(-theta * dt)
                    # Transition following constant mean mu0
                    mu = mu0 + phi * (x_prev - mu0)
                    ou_var = (sigma0**2) * (1 - jnp.exp(-2 * theta * dt))
                    xt = numpyro.sample("xt", dist.Normal(mu, jnp.sqrt(jnp.maximum(ou_var, 1e-10))))
                    return xt, xt

                _, x_path = scan(transition_fn, init=x0, xs=(subject_t[:-1], subject_t[1:]))
                x_all.append(jnp.concatenate([x0[None, :], x_path], axis=0))

        x_flat = jnp.concatenate(x_all, axis=0)
        numpyro.deterministic("x_trace", x_flat)
        
        # Likelihood
        mu_obs = jnp.matmul(x_flat, Lambda.T)
        with numpyro.plate("data", N_total):
            # .to_event(1) handles multidimensional observations
            numpyro.sample("y", dist.Normal(mu_obs, sigma_obs).to_event(1), obs=y)

    def fit(self, rng_key, Nsub, N_total, start_idx, end_idx, t, y, num_warmup=1000, num_samples=1000, **kwargs):
        # Ignore extra kwargs (like s_cov) used by CGSLOU
        input_data = {"Nsub": Nsub, "N_total": N_total, "start_idx": start_idx, 
                      "end_idx": end_idx, "t": t, "y": y}
        kernel = NUTS(self.model, target_accept_prob=self.target_accept_prob)
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
        self.mcmc.run(rng_key, **input_data)
        return self.mcmc.get_samples()
    

class CGSLOU:
    def __init__(self, K, D, target_accept_prob=0.95):
        self.K = K  
        self.D = D  
        self.target_accept_prob = target_accept_prob
        self.mcmc = None
        self.last_input_data = None
        
    def model(self, Nsub, N_total, start_idx, end_idx, t, y, s_cov=None):
        """
        Modified to include:
        1. Time-varying mean: mu(s_i, t) = alpha + Phi * s_i * t
        2. Global-Local Shrinkage Prior for Lambda
        """
        # --- 1. OU Process Parameters ---
        # Theta: Reversion speed (Diagonal of matrix Theta in image)
        theta_diag = numpyro.sample("theta_diag", dist.HalfNormal(1.0).expand([self.K]))
        # alpha: Intercept of the mean process
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0).expand([self.K]))
        # phi_latent: Slope for s_i * t (corresponds to Phi in image)
        phi_latent = numpyro.sample("phi_latent", dist.Normal(0.0, 2.0).expand([self.K]))
        # gamma: Diffusion volatility (Gamma in image)
        gamma = numpyro.sample("gamma", dist.HalfCauchy(2.5).expand([self.K]))
        
        # # --- 2. Global-Local Shrinkage Prior for Lambda ---
        # # Auxiliary variables for Inverse-Gamma mixture
        # eta_global = numpyro.sample("eta_global", dist.InverseGamma(0.5, 1.0).expand([self.K]))
        # tau_sq = numpyro.sample("tau_sq", dist.InverseGamma(0.5, 1.0 / eta_global)) # Global Scale
        
        # nu_local = numpyro.sample("nu_local", dist.InverseGamma(0.5, 1.0).expand([self.D, self.K]))
        # xi_sq = numpyro.sample("xi_sq", dist.InverseGamma(0.5, 1.0 / nu_local)) # Local Scale
        
        # # Non-centered parameterization for Lambda
        # z_lambda = numpyro.sample("z_lambda", dist.Normal(0.0, 1.0).expand([self.D, self.K]))
        # Lambda = numpyro.deterministic("Lambda", z_lambda * jnp.sqrt(tau_sq * xi_sq))

        # --- Global-Local Shrinkage Prior for Lambda with Identifiability ---
        eta_global = numpyro.sample("eta_global", dist.InverseGamma(0.5, 1.0).expand([self.K]))
        tau_sq = numpyro.sample("tau_sq", dist.InverseGamma(0.5, 1.0 / eta_global)) 
        
        nu_local = numpyro.sample("nu_local", dist.InverseGamma(0.5, 1.0).expand([self.D, self.K]))
        xi_sq = numpyro.sample("xi_sq", dist.InverseGamma(0.5, 1.0 / nu_local)) 
        
        # Non-centered parameterization
        z_lambda = numpyro.sample("z_lambda", dist.Normal(0.0, 1.0).expand([self.D, self.K]))
        
        # Calculate raw Lambda with shrinkage
        Lambda_raw = z_lambda * jnp.sqrt(tau_sq * xi_sq)

        # Apply constraints for identifiability:
        # 1. Lower triangular: Lambda[d, k] = 0 if k > d
        # 2. Positive diagonal: Lambda[k, k] > 0
        mask = jnp.tril(jnp.ones((self.D, self.K)))
        Lambda_tri = Lambda_raw * mask
        
        # Ensure diagonal elements are positive to fix sign-flipping
        diag_idx = jnp.arange(self.K)
        Lambda = Lambda_tri.at[diag_idx, diag_idx].set(jnp.abs(Lambda_tri[diag_idx, diag_idx]))
        numpyro.deterministic("Lambda", Lambda)

        # --- 3. Observation Noise (Psi) ---
        psi_diag = numpyro.sample("psi_diag", dist.HalfCauchy(2.5).expand([self.D]))

        # --- 4. Latent Dynamics (Vectorized per Subject) ---
        # Define the fixed number of steps per subject
        # Assuming all subjects have the same number of observations 'T'
        # If they vary, you must pad them to a max length and use masking.
        steps_per_sub = N_total // Nsub 

        x_all = []
        for i in range(Nsub):
            with numpyro.handlers.scope(prefix=f"sub_{i}"):
                # 1. Use lax.dynamic_slice for JAX-compatible indexing
                # slice(start_index, slice_size)
                s_i = start_idx[i]
                
                # Dynamic slice for time t and observations y
                subject_t = lax.dynamic_slice(t, (s_i,), (steps_per_sub,))
                subject_y = lax.dynamic_slice(y, (s_i, 0), (steps_per_sub, self.D))
                
                subject_s = s_cov[i] if s_cov is not None else 0.0
                
                # Initial state logic
                mu_t0 = alpha + phi_latent * subject_s * subject_t[0]
                x0 = numpyro.sample("x_init", dist.Normal(mu_t0, 1.0))
                
                def transition_fn(carry, inputs):
                    x_prev = carry
                    t_p, t_c = inputs
                    dt = t_c - t_p
                    mu_p = alpha + phi_latent * subject_s * t_p
                    mu_c = alpha + phi_latent * subject_s * t_c
                    phi = jnp.exp(-theta_diag * dt)
                    mu_xt = mu_c + phi * (x_prev - mu_p)
                    var_xt = (gamma**2 / (2 * theta_diag)) * (1 - jnp.exp(-2 * theta_diag * dt))
                    xt = numpyro.sample("xt", dist.Normal(mu_xt, jnp.sqrt(jnp.maximum(var_xt, 1e-10))))
                    return xt, xt

                # 2. Run scan with the statically-sized slices
                _, x_path = scan(transition_fn, x0, (subject_t[:-1], subject_t[1:]))
                x_subject = jnp.concatenate([x0[None, :], x_path], axis=0)
                x_all.append(x_subject)

        x_flat = jnp.concatenate(x_all, axis=0)
        numpyro.deterministic("x_trace", x_flat)
        
        # --- 5. Likelihood ---
        # mu_obs shape: (N_total, D)
        mu_obs = jnp.matmul(x_flat, Lambda.T)
        
        # .to_event(1) makes the D-dimension an 'event' dimension
        # instead of a 'batch' dimension, which resolves the broadcasting conflict.
        with numpyro.plate("data", N_total):
            numpyro.sample("y", dist.Normal(mu_obs, psi_diag).to_event(1), obs=y)

    def fit(self, rng_key, Nsub, N_total, start_idx, end_idx, t, y, s_cov=None,
            num_samples=1000, num_warmup=1000, num_chains=1):
        
        # Initialize s_cov as zeros if not provided
        if s_cov is None:
            s_cov = jnp.zeros(Nsub)

        self.last_input_data = {
            "Nsub": Nsub, "N_total": N_total, 
            "start_idx": start_idx, "end_idx": end_idx, 
            "t": t, "y": y, "s_cov": s_cov
        }

        kernel = NUTS(self.model, target_accept_prob=self.target_accept_prob)
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        self.mcmc.run(rng_key, **self.last_input_data)
        return self.mcmc.get_samples()
