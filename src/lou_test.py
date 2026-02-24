import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from numpyro.contrib.control_flow import scan


class OULatentModel:
    def __init__(self, K, D, target_accept_prob=0.95):
        self.K, self.D = K, D
        self.target_accept_prob = target_accept_prob

    def model(self, N_total, t, y, is_start):
        # Priors
        theta = numpyro.sample("theta", dist.LogNormal(0.0, 0.5).expand([self.K]))
        mu0 = numpyro.sample("mu0", dist.Normal(0.0, 1.0).expand([self.K]))
        sigma0 = numpyro.sample("sigma0", dist.HalfNormal(1.0).expand([self.K]))
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.5))
        Lambda = numpyro.sample("Lambda", dist.Normal(0.0, 1.0).expand([self.D, self.K]))

        def transition_fn(carry, i):
            x_prev, t_prev = carry
            dt = jnp.clip(t[i] - t_prev, a_min=1e-5)
            phi = jnp.exp(-theta * dt)
            ou_sd = sigma0 * jnp.sqrt(jnp.clip((1 - phi**2) / (2 * theta), a_min=1e-9))
            
            # Reset logic for new subjects
            mu_curr = jnp.where(is_start[i], mu0, mu0 + phi * (x_prev - mu0))
            sd_curr = jnp.where(is_start[i], sigma0, ou_sd)
            
            x_curr = numpyro.sample("x", dist.Normal(mu_curr, sd_curr))
            return (x_curr, t[i]), x_curr

        _, x_samples = scan(transition_fn, (jnp.zeros(self.K), 0.0), jnp.arange(N_total))

        with numpyro.plate("obs", N_total):
            numpyro.sample("y", dist.MultivariateNormal(x_samples @ Lambda.T, 
                                                        jnp.eye(self.D) * sigma_obs**2), obs=y)

    def fit(self, rng_key, data_dict, num_warmup=500, num_samples=500):
        nuts_kernel = NUTS(self.model, target_accept_prob=self.target_accept_prob)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
        mcmc.run(rng_key, **data_dict)
        return mcmc.get_samples()

def simulate_data(Nsub=10, K=2, D=5):
    """Generates synthetic data for testing."""
    np.random.seed(123)
    sigma_obs_true = 0.2
    theta_true = np.array([0.5, 1.0])
    mu0_true = np.array([0.0, 1.0])
    sigma0_true = np.array([0.3, 0.4])
    Lambda_true = np.array([
        [1.0, 0.5], [0.3, 1.2], [0.0, 0.8], [1.0, 0.3], [0.2, 0.7]
    ])

    y_list, t_list, x_list = [], [], []
    for i in range(Nsub):
        T_i = np.random.randint(5, 11)
        t_i = np.concatenate([[0.0], np.cumsum(np.random.uniform(0.5, 1.5, size=T_i - 1))])
        x_i = np.zeros((T_i, K))
        x_i[0] = np.random.normal(mu0_true, sigma0_true)

        for n in range(1, T_i):
            phi = np.exp(-theta_true * (t_i[n] - t_i[n - 1]))
            ou_sd = sigma0_true * np.sqrt((1 - phi ** 2) / (2 * theta_true))
            mu = mu0_true + phi * (x_i[n - 1] - mu0_true)
            x_i[n] = np.random.normal(mu, ou_sd)

        y_i = x_i @ Lambda_true.T + np.random.normal(0, sigma_obs_true, size=(T_i, D))
        x_list.append(x_i); y_list.append(y_i); t_list.append(t_i)

    start_idx = np.cumsum([0] + [len(y) for y in y_list[:-1]])
    end_idx = start_idx + np.array([len(y) for y in y_list]) - 1

    return {
        'Nsub': Nsub, 'N_total': sum(len(y) for y in y_list),
        'start_idx': start_idx, 'end_idx': end_idx,
        't': jnp.array(np.concatenate(t_list)), 'y': jnp.array(np.vstack(y_list)),
        'ground_truth': {
            'theta': theta_true, 'mu0': mu0_true, 
            'sigma0': sigma0_true, 'sigma_obs': sigma_obs_true,
            'Lambda': Lambda_true
        }
    }

if __name__ == "__main__":
    # Parameters
    K, D = 2, 5
    rng_key = jax.random.PRNGKey(123)
    
    # 1. Simulate Data
    print("Simulating data...")
    data = simulate_data(Nsub=10, K=K, D=D)
    
    # 2. Initialize and Fit Class-based Model
    # Lambda is now an inferred random variable inside the model
    print("Starting MCMC inference (inferring Lambda)...")
    model_instance = OULatentModel(K=K, D=D)
    samples = model_instance.fit(rng_key, data, num_warmup=500, num_samples=500)
    
    # 3. Results Summary
    def summarize(name):
        s = samples[name]
        return jnp.mean(s, axis=0), jnp.std(s, axis=0)

    print("\n--- Parameter Estimates ---")
    for param in ["theta", "mu0", "sigma0", "sigma_obs"]:
        mean, std = summarize(param)
        print(f"{param}: Mean={mean}, Std={std} (True={data['ground_truth'][param]})")

    print("\n--- Lambda (Factor Loadings) Estimate ---")
    lambda_mean, _ = summarize("Lambda")
    print("Estimated Lambda Mean:\n", lambda_mean)
    print("True Lambda:\n", data['ground_truth']['Lambda'])

    # 4. Latent States Visualization
    N_total = data['N_total']
    x_post = jnp.mean(jnp.stack([samples[f"x_{n}"] for n in range(N_total)], axis=1), axis=0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(x_post[:, 0], label="Latent Dim 1 (Post. Mean)")
    plt.plot(x_post[:, 1], label="Latent Dim 2 (Post. Mean)")
    plt.title("Inferred Latent States Over Time")
    plt.legend(); plt.show()