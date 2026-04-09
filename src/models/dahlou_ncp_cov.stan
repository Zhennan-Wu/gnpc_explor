functions {
  // Helper to solve the Continuous Lyapunov Equation: Gamma * Omega + Omega * Gamma' = Sigma
  matrix solve_lyapunov(matrix Gamma, matrix Sigma, int R) {
    matrix[R*R, R*R] K;
    vector[R*R] vec_Sigma;
    vector[R*R] vec_Omega;
    matrix[R, R] Omega;
    
    // Construct Kronecker sum K = (I \otimes Gamma) + (Gamma \otimes I)
    for (i in 1:R) {
      for (j in 1:R) {
        for (k in 1:R) {
          for (l in 1:R) {
            int row_idx = (j - 1) * R + i;
            int col_idx = (l - 1) * R + k;
            K[row_idx, col_idx] = (i == k ? Gamma[j, l] : 0.0) + (j == l ? Gamma[i, k] : 0.0);
          }
        }
      }
    }
    
    for (i in 1:R) {
      for (j in 1:R) {
        vec_Sigma[(j - 1) * R + i] = Sigma[i, j];
      }
    }
    
    vec_Omega = K \ vec_Sigma;
    
    for (i in 1:R) {
      for (j in 1:R) {
        Omega[i, j] = vec_Omega[(j - 1) * R + i];
      }
    }
    return Omega;
  }
}

data {
    int<lower=1> N;         // Total number of observations
    int<lower=1> Nsub;      // Number of individuals
    int<lower=1> K;         // Number of items (12)
    int<lower=1> R;         // Number of latent factors (4)
    int<lower=1> p_dyn;     // Number of dynamic covariates (3)
    int<lower=1> p_meas;    // Number of measurement covariates (1)

    array[N] int<lower=1, upper=Nsub> ID;
    array[Nsub] int cumu;
    array[Nsub] int repme;
    
    // Y must be 1-5 (ALSFRS 0-4 + 1)
    array[N, K] int<lower=1, upper=5> Y;

    vector[N] deltat;
    vector[N] t_abs;     // NEW: Absolute time elapsed since baseline
    matrix[N, p_dyn] X_dyn;
    matrix[N, p_meas] X_meas;
}

parameters {
    // --- IRT Measurement Parameters ---
    // 12 items, 5 categories each -> 4 thresholds per item
    array[K] ordered[4] theta; 

    // 12 total loadings, 4 are anchored at 1.0, leaving 8 free
    vector<lower=1e-6>[K - R] lambda_free;
    real<lower=1e-6> sigma_lambda;

    // Covariate Effects
    matrix[K, p_meas] beta;       // Uric acid acting on the items directly
    matrix[R, p_dyn] alpha_dyn;   // Treatment/Delay acting on the latent disease state

    matrix[Nsub, K] b_raw;
    vector<lower=1e-6>[K] sigma_bk;

    // --- Structural Parameters ---
    cholesky_factor_corr[R] L_S_corr;     
    vector<lower=0>[R] L_S_scale;         
    
    vector[R * (R - 1) / 2] a_low;
    
    cholesky_factor_corr[R] L_Sigma_corr; 
    vector<lower=0>[R] L_Sigma_scale;     
    
    // --- Latent States (Non-Centered GMRF) ---
    matrix[R, N] xi_raw;
}

transformed parameters {
    matrix[Nsub, K] b;
    vector[K] lambda; 
    
    // Structural Matrices
    matrix[R, R] L_S;
    matrix[R, R] S;
    matrix[R, R] A;
    matrix[R, R] Gamma;
    
    matrix[R, R] L_Sigma;
    matrix[R, R] Sigma;
    matrix[R, R] Omega;
    matrix[R, N] xi;

    // 1. Construct Full Loadings Vector (Anchored per Domain)
    // Bulbar
    lambda[1] = 1.0;              
    lambda[2] = lambda_free[1];
    lambda[3] = lambda_free[2];
    
    // Fine Motor
    lambda[4] = 1.0;              
    lambda[5] = lambda_free[3];
    lambda[6] = lambda_free[4];
    
    // Gross Motor
    lambda[7] = 1.0;              
    lambda[8] = lambda_free[5];
    lambda[9] = lambda_free[6];
    
    // Respiratory
    lambda[10] = 1.0;              
    lambda[11] = lambda_free[7];
    lambda[12] = lambda_free[8];

    // 2. Calculate random effects
    for (i in 1:Nsub){
        for (k in 1:K){
            b[i, k] = b_raw[i, k] * sigma_bk[k];
        }
    }
    
    // 3. Construct stable Gamma = S + A
    L_S = diag_pre_multiply(L_S_scale, L_S_corr);
    S = multiply_lower_tri_self_transpose(L_S);
    
    A = rep_matrix(0, R, R);
    {
      int pos = 1;
      for (i in 2:R) {
        for (j in 1:(i-1)) {
          A[i, j] = a_low[pos];
          A[j, i] = -a_low[pos];
          pos += 1;
        }
      }
    }
    Gamma = S + A;

    // 4. Construct diffusion matrix and solve Lyapunov for Omega
    L_Sigma = diag_pre_multiply(L_Sigma_scale, L_Sigma_corr);
    Sigma = multiply_lower_tri_self_transpose(L_Sigma);
    
    Omega = solve_lyapunov(Gamma, Sigma, R);
    Omega = 0.5 * (Omega + Omega'); 
    Omega = add_diag(Omega, 1e-5);

    // 5. Generate Latent Trajectories with Dynamic Covariates
    {
        matrix[R, R] L_Omega = cholesky_decompose(Omega);
        for (i in 1:Nsub) {
            int start_idx = cumu[i] - repme[i] + 1;
            
            // Time 1: Scaled by absolute time (usually 0 at baseline)
            vector[R] mu_start = (alpha_dyn * X_dyn[start_idx, ]') * t_abs[start_idx];
            xi[:, start_idx] = mu_start + L_Omega * xi_raw[:, start_idx];
            
            for (j in 2:repme[i]) {
                int k = start_idx + j - 1;
                matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
                
                matrix[R, R] Q = Omega - Phi * Omega * Phi';
                matrix[R, R] Q_sym = 0.5 * (Q + Q'); 
                matrix[R, R] L_Q = cholesky_decompose(add_diag(Q_sym, 1e-6));
                
                // Times k and k-1: Scaled by absolute time
                vector[R] mu_k = (alpha_dyn * X_dyn[k, ]') * t_abs[k];
                vector[R] mu_prev = (alpha_dyn * X_dyn[k-1, ]') * t_abs[k-1];
                
                xi[:, k] = mu_k + Phi * (xi[:, k-1] - mu_prev) + L_Q * xi_raw[:, k];
            }
        }
    }
}

model {
    // Priors (IRT Cutpoints)
    for (k in 1:K) {
        theta[k] ~ normal(0, 5); 
    }
    
    lambda_free ~ normal(1, sigma_lambda);
    sigma_lambda ~ cauchy(0, 5);
    
    // Priors for covariates
    to_vector(alpha_dyn) ~ normal(0, 5);
    to_vector(beta) ~ cauchy(0, 5);
    
    sigma_bk ~ cauchy(0, 5);
    to_vector(b_raw) ~ normal(0, 1);
    
    // Priors (Structural & Latent)
    L_S_corr ~ lkj_corr_cholesky(2.0);
    L_S_scale ~ lognormal(0, 0.5); 
    
    a_low ~ normal(0, 0.5);
    
    L_Sigma_corr ~ lkj_corr_cholesky(2.0);
    L_Sigma_scale ~ lognormal(0, 0.5);
    
    to_vector(xi_raw) ~ std_normal();

    // --- Likelihood (IRT) ---
    for (i in 1:N) {
        int sub = ID[i];
        row_vector[p_meas] Xi_m = X_meas[i, ];
        
        // Bulbar Domain
        Y[i, 1] ~ ordered_logistic(Xi_m * beta[1, ]' + lambda[1] * xi[1, i] + b[sub, 1], theta[1]);
        Y[i, 2] ~ ordered_logistic(Xi_m * beta[2, ]' + lambda[2] * xi[1, i] + b[sub, 2], theta[2]);
        Y[i, 3] ~ ordered_logistic(Xi_m * beta[3, ]' + lambda[3] * xi[1, i] + b[sub, 3], theta[3]);
        
        // Fine Motor Domain
        Y[i, 4] ~ ordered_logistic(Xi_m * beta[4, ]' + lambda[4] * xi[2, i] + b[sub, 4], theta[4]);
        Y[i, 5] ~ ordered_logistic(Xi_m * beta[5, ]' + lambda[5] * xi[2, i] + b[sub, 5], theta[5]);
        Y[i, 6] ~ ordered_logistic(Xi_m * beta[6, ]' + lambda[6] * xi[2, i] + b[sub, 6], theta[6]);
        
        // Gross Motor Domain
        Y[i, 7] ~ ordered_logistic(Xi_m * beta[7, ]' + lambda[7] * xi[3, i] + b[sub, 7], theta[7]);
        Y[i, 8] ~ ordered_logistic(Xi_m * beta[8, ]' + lambda[8] * xi[3, i] + b[sub, 8], theta[8]);
        Y[i, 9] ~ ordered_logistic(Xi_m * beta[9, ]' + lambda[9] * xi[3, i] + b[sub, 9], theta[9]);
        
        // Respiratory Domain
        Y[i, 10] ~ ordered_logistic(Xi_m * beta[10, ]' + lambda[10] * xi[4, i] + b[sub, 10], theta[10]);
        Y[i, 11] ~ ordered_logistic(Xi_m * beta[11, ]' + lambda[11] * xi[4, i] + b[sub, 11], theta[11]);
        Y[i, 12] ~ ordered_logistic(Xi_m * beta[12, ]' + lambda[12] * xi[4, i] + b[sub, 12], theta[12]);
    }
}