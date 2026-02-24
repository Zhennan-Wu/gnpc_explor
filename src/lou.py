import jax
import jax.numpy as jnp
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive


class LOU:
    def __init__(self, K, D, target_accept_prob=0.95):
        self.K = K  # Latent dimension (e.g., 2)
        self.D = D  # Observed dimension (e.g., 10)
        self.target_accept_prob = target_accept_prob
        self.mcmc = None
        
    def model(self, Nsub, N_total, start_idx, end_idx, t, y):
        # --- Priors for OU Dynamics ---
        theta = numpyro.sample("theta", dist.LogNormal(jnp.log(1.0), 0.3).expand([self.K]))
        mu0 = numpyro.sample("mu0", dist.Normal(0.0, 2.0).expand([self.K]))
        sigma0 = numpyro.sample("sigma0", dist.HalfNormal(0.5).expand([self.K]))
        
        # --- Priors for Observation Noise ---
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.2))

        # --- Prior for Loading Matrix (Lambda) ---
        # Shape is (D, K). We use a Normal prior for each entry.
        # You can adjust the scale (0.5) based on your expected signal strength.
        Lambda = numpyro.sample(
            "Lambda", 
            dist.Normal(0.0, 0.5).expand([self.D, self.K])
        )

        # --- Latent Dynamics (Vectorized per Subject) ---
        x_all = []
        for i in range(Nsub):
            s, e = start_idx[i], end_idx[i]
            
            # Initial state for subject i
            x0 = numpyro.sample(f"x_init_{i}", dist.Normal(mu0, sigma0))
            
            def transition_fn(carry, inputs):
                x_prev = carry
                t_prev, t_curr = inputs
                dt = t_curr - t_prev
                
                phi = jnp.exp(-theta * dt)
                # Numerical stability: ensure variance is positive
                ou_var = (sigma0**2) * (1 - jnp.exp(-2 * theta * dt))
                ou_sd = jnp.sqrt(jnp.maximum(ou_var, 1e-10))
                
                mu = mu0 + phi * (x_prev - mu0)
                xt = numpyro.sample("xt", dist.Normal(mu, ou_sd))
                return xt, xt

            subject_t = t[s:e+1]
            _, x_path = scan(
                    transition_fn, 
                    init=x0, 
                    xs=(subject_t[:-1], subject_t[1:]),
                    history_key=f"subject_{i}"  # <--- Add this line
                )
            x_subject = jnp.concatenate([x0[None, :], x_path], axis=0)
            x_all.append(x_subject)

        x_flat = jnp.concatenate(x_all, axis=0)
        numpyro.deterministic("x_trace", x_flat)
        
        # --- Observation Likelihood ---
        # mu_obs = x_flat @ Lambda^T -> Shape (N_total, D)
        mu_obs = jnp.matmul(x_flat, Lambda.T)
        
        with numpyro.plate("data", N_total):
            # Using Normal with expand to handle D dimensions efficiently
            numpyro.sample("y", dist.Normal(mu_obs, sigma_obs), obs=y)

    def fit(self, rng_key, Nsub, N_total, start_idx, end_idx, t, y, 
            num_samples=1000, num_warmup=1000, num_chains=1):

        # Store for the Visualizer's Predictive step
        self.last_input_data = {
            "Nsub": Nsub, "N_total": N_total, 
            "start_idx": start_idx, "end_idx": end_idx, "t": t, "y": y
        }

        kernel = NUTS(self.model, target_accept_prob=self.target_accept_prob)
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        
        # Note: Lambda is no longer passed as an argument; it's inferred
        self.mcmc.run(rng_key, Nsub, N_total, start_idx, end_idx, t, y)
        return self.mcmc.get_samples()
    
    def get_latent_trajectories(self, summary=True):
            """
            Extracts and reconstructs the latent states (x) for all subjects.
            
            Args:
                summary (bool): If True, returns the posterior mean (N_total, K).
                                If False, returns all samples (num_samples, N_total, K).
            """
            if self.mcmc is None:
                raise ValueError("Model must be fitted first.")
                
            samples = self.mcmc.get_samples()
            num_samples = samples['theta'].shape[0]
            
            # 1. Identify how many subjects we have by looking for 'x_init_i' keys
            subject_init_keys = sorted([k for k in samples.keys() if k.startswith("x_init_")])
            Nsub = len(subject_init_keys)
            
            # 2. Extract 'xt' which contains the scanned transitions. 
            # In NumPyro scan, 'xt' will have shape (num_samples, N_steps_per_subject, K)
            # However, because we scanned per subject in a loop, NumPyro might have 
            # individual sites if not handled carefully. 
            # Assuming the model used the `scan` logic provided previously:
            
            all_latent_samples = []
            
            for i in range(Nsub):
                # Get initial state for this subject: (num_samples, K)
                x0 = samples[f"x_init_{i}"][:, jnp.newaxis, :] 
                
                # Get the rest of the path for this subject. 
                # If using the 'scan' inside a loop, we need to ensure we grab the 
                # correct 'xt' or equivalent site.
                # Note: In the previous 'scan' implementation, the site was named "xt"
                # But in a loop, NumPyro typically suffixes them (e.g., "xt", "xt_1", etc.)
                
                # For simplicity, we assume the user wants the flattened x_flat 
                # that we calculated in the model. Since we didn't explicitly 
                # 'deterministic' that, let's add it to the model or reconstruct it.
                pass

            # REVISED APPROACH: Re-run the model with Predictive or 
            # rely on the fact that we can reconstruct it from the samples.
            
            # A cleaner way is to use numpyro.infer.Predictive to get 'x' directly
            
            # We need a dummy run of the model to capture the 'x_flat' 
            # if we wrap it in numpyro.deterministic
            predictive = Predictive(self.model, samples)
            # We pass the same data used for fitting
            # Note: we need to ensure 'x_flat' was marked as deterministic in the model
            
            # Let's assume we modify the model to include: 
            # numpyro.deterministic("x_trace", x_flat)
            
            full_samples = predictive(
                jax.random.PRNGKey(1), 
                **self.last_input_data # You should store inputs in .fit()
            )
            
            x_trace = full_samples["x_trace"] # Shape: (num_samples, N_total, K)
            
            if summary:
                return jnp.mean(x_trace, axis=0)
            return x_trace