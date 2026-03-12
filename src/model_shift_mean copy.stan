// Model 2a: Latent Factor Mean as a function of Covariates and Time
data {
    int<lower=1> N;                   
    int<lower=1> Nsub;                
    int<lower=1> K;                   
    int<lower=1> R;                   
    int<lower=1> p;                   // Number of item-level covariates
    int<lower=1> q;                   // Number of latent-level covariates (including time)

    array[N] int<lower=1, upper=Nsub> ID;
    array[Nsub] int cumu;             
    array[Nsub] int repme;            

    array[N, K] int Y;                
    array[N, K] int missing_ID;       

    vector[N] deltat;
    vector[N] time;                   // Absolute time for mean function
    matrix[N, p] X;                   // Item covariates
    matrix[N, q] Z;                   // Latent mean covariates (e.g., age, gender, etc.)

    int<lower=2> ncate4;
    int<lower=2> ncate5;
    int<lower=2> ncate6;
    int<lower=2> ncate7;
}

parameters {
    real theta1; real theta2; real theta3;

    ordered[ncate4 - 1] theta4;
    ordered[ncate5 - 1] theta5;
    ordered[ncate6 - 1] theta6;
    ordered[ncate7 - 1] theta7;

    real mu_theta;
    real<lower=0> sigma_theta;

    vector<lower=0>[K] lambda;
    real<lower=0> sigma_lambda;

    matrix[K, p] beta;                // Item regression
    matrix[R, q] gamma_latent;        // Latent mean regression (Effect of Z on Xi)

    matrix[Nsub, K] b_raw;
    vector<lower=0>[K] sigma_bk;

    array[N] vector[R] xi;           
    matrix[R, R] Gamma;              
    real<lower=-1, upper=1> rho;     
}

transformed parameters {
    matrix[Nsub, K] b;
    corr_matrix[R] Omega;

    Omega[1, 1] = 1.0; 
    Omega[2, 2] = 1.0;
    Omega[1, 2] = rho; 
    Omega[2, 1] = rho;

    for (i in 1:Nsub) {
        for (k in 1:K) {
            b[i, k] = b_raw[i, k] * sigma_bk[k];
        }
    }
}

model {
    // Priors
    mu_theta ~ normal(0, 10);
    sigma_theta ~ cauchy(0, 5);
    lambda ~ normal(1, sigma_lambda);
    to_vector(beta) ~ cauchy(0, 5);
    to_vector(gamma_latent) ~ normal(0, 2); // Priors for latent mean coefficients
    to_vector(Gamma) ~ normal(0, 5);

    // Latent Factor Dynamics with Mean Structure
    for (i in 1:Nsub) {
        int start_idx = cumu[i] - repme[i] + 1;
        
        // Initial state mean depends on Z at time 1
        vector[R] mu_start = gamma_latent * Z[start_idx]' * time[start_idx];
        xi[start_idx] ~ multi_normal(mu_start, Omega);

        for (j in 2:repme[i]) {
            int k = start_idx + j - 1;
            matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
            matrix[R, R] Q = Omega - Phi * Omega * Phi';
            matrix[R, R] Q_sym = 0.5 * (Q + Q');
            
            // The mean is the autoregressive part PLUS the covariate effect
            // We use the 'steady-state' logic where the mean shifts toward gamma_latent * Z
            vector[R] target_k = gamma_latent * Z[k]'*time[k];
            vector[R] target_prev = (gamma_latent * Z[k-1]') * time[k-1];
            vector[R] cond_mean = target_k + Phi * (xi[k-1] - target_prev);
            
            xi[k] ~ multi_normal(cond_mean, add_diag(Q_sym, 1e-9));
        }
    }

    // Likelihood (same as before)
    for (i in 1:N) {
        row_vector[p] curr_X = X[i, ];
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + curr_X * beta[1, ]' + lambda[1] * xi[i, 1] + b[ID[i], 1]);
        if (missing_ID[i, 4] == 0) Y[i, 4] ~ ordered_logistic(curr_X * beta[4, ]' + lambda[4] * xi[i, 2] + b[ID[i], 4], theta4);
        // ... (repeat for other items)
    }
}
