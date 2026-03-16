// alsfrs_ou_4factor.stan
data {
    int<lower=1> N;
    int<lower=1> Nsub;
    int<lower=1> K;
    int<lower=1> D;

    array[N] int<lower=1, upper=Nsub> ID;
    vector[N] deltat;
    array[N] int<lower=0, upper=1> is_first_visit;
    
    array[N, K] int<lower=0, upper=5> Y; // 0 = Missing, 1-5 = Ordinal Scores
    array[K] int<lower=1, upper=D> factor_map;
}

parameters {
    array[K] ordered[4] cutpoints;       // 4 cutpoints for 5 categories (0 to 4 score + 1)
    vector<lower=0>[K] lambda;           // Factor loadings
    
    vector<lower=0>[D] gamma;            // Rate of mean reversion
    cholesky_factor_corr[D] L_Omega;     // Cholesky correlation matrix for stationary distribution
    
    matrix[N, D] xi;                     // Latent traits (N rows, D factors)
}

transformed parameters {
    // Reconstruct full correlation matrix for the conditional transition
    matrix[D, D] Omega = multiply_lower_tri_self_transpose(L_Omega);
}

model {
    // Priors
    for (k in 1:K) {
        cutpoints[k] ~ normal(0, 3);
        lambda[k] ~ lognormal(0, 1);
    }
    gamma ~ lognormal(0, 1);
    L_Omega ~ lkj_corr_cholesky(2.0);
    
    // Latent OU Process Evolution
    for (i in 1:N) {
        if (is_first_visit[i] == 1) {
            // Stationary distribution at baseline (mean 0, variance 1 for identifiability)
            xi[i, ] ~ multi_normal_cholesky(rep_vector(0.0, D), L_Omega);
        } else {
            vector[D] mean_t;
            matrix[D, D] cov_t;
            
            // Construct the exact OU transition mean and covariance
            for (d in 1:D) {
                real exp_d = exp(-gamma[d] * deltat[i]);
                mean_t[d] = exp_d * xi[i-1, d]; 
                
                for (d2 in 1:D) {
                    real exp_d2 = exp(-gamma[d2] * deltat[i]);
                    // Covariance scales with time
                    cov_t[d, d2] = Omega[d, d2] * (1.0 - exp_d * exp_d2);
                }
            }
            
            // Add tiny numerical jitter to the diagonal to ensure positive definiteness
            for (d in 1:D) {
                cov_t[d, d] += 1e-6; 
            }
            
            xi[i, ] ~ multi_normal(mean_t, cov_t);
        }
    }
    
    // Measurement Model (Ordinal IRT)
    for (i in 1:N) {
        for (k in 1:K) {
            if (Y[i, k] > 0) { // Skip missing data
                Y[i, k] ~ ordered_logistic(lambda[k] * xi[i, factor_map[k]], cutpoints[k]);
            }
        }
    }
}