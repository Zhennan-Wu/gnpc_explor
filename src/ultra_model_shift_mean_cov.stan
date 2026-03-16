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
    int<lower=1> K;         // Number of items
    int<lower=1> R;         // Number of latent factors (e.g., 2)
    int<lower=1> p;         // Number of covariates

    array[N] int<lower=1, upper=Nsub> ID;
    array[Nsub] int cumu;
    array[Nsub] int repme;

    array[N, K] int Y;
    array[N, K] int missing_ID;

    vector[N] deltat;
    matrix[N, p] X;

    int<lower=2> ncate4;
    int<lower=2> ncate5;
    int<lower=2> ncate6;
    int<lower=2> ncate7;
}

parameters {
    // --- IRT Measurement Parameters ---
    real theta1; 
    real theta2; 
    real theta3;
    ordered[ncate4 - 1] theta4;
    ordered[ncate5 - 1] theta5;
    ordered[ncate6 - 1] theta6;
    ordered[ncate7 - 1] theta7;

    real mu_theta;
    real<lower=1e-6> sigma_theta;

    vector<lower=1e-6>[K - 2] lambda_free;
    real<lower=1e-6> sigma_lambda;

    // Measurement Covariate Effects
    matrix[K, p] beta;

    matrix[Nsub, K] b_raw;
    vector<lower=1e-6>[K] sigma_bk;

    // --- Dynamic Covariate Parameters (NEW) ---
    // Effects of covariates on the equilibrium mean of the latent states
    matrix[R, p] alpha_dyn; 

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

    // 1. Construct Full Loadings Vector (Anchored)
    lambda[1] = 1.0;              
    lambda[2] = lambda_free[1];
    lambda[3] = lambda_free[2];
    
    lambda[4] = 1.0;              
    lambda[5] = lambda_free[3];
    lambda[6] = lambda_free[4];
    lambda[7] = lambda_free[5];

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
            
            // Calculate stationary mean for Time 1 based on covariates
            vector[R] mu_start = alpha_dyn * X[start_idx, ]';
            
            // Time 1: Shifted by mu_start
            xi[:, start_idx] = mu_start + L_Omega * xi_raw[:, start_idx];
            
            for (j in 2:repme[i]) {
                int k = start_idx + j - 1;
                matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
                
                matrix[R, R] Q = Omega - Phi * Omega * Phi';
                matrix[R, R] Q_sym = 0.5 * (Q + Q'); 
                matrix[R, R] L_Q = cholesky_decompose(add_diag(Q_sym, 1e-6));
                
                // Calculate moving equilibrium for current and previous time points
                vector[R] mu_k = alpha_dyn * X[k, ]';
                vector[R] mu_prev = alpha_dyn * X[k-1, ]';
                
                // Deterministic mapping: reverts toward the dynamic moving average
                xi[:, k] = mu_k + Phi * (xi[:, k-1] - mu_prev) + L_Q * xi_raw[:, k];
            }
        }
    }
}

model {
    // Priors (IRT)
    theta1 ~ normal(mu_theta, sigma_theta);
    theta2 ~ normal(mu_theta, sigma_theta);
    theta3 ~ normal(mu_theta, sigma_theta);
    theta4 ~ normal(mu_theta, sigma_theta);
    theta5 ~ normal(mu_theta, sigma_theta);
    theta6 ~ normal(mu_theta, sigma_theta);
    theta7 ~ normal(mu_theta, sigma_theta);

    mu_theta ~ normal(0, 10);
    sigma_theta ~ cauchy(0, 5);
    
    lambda_free ~ normal(1, sigma_lambda);
    sigma_lambda ~ cauchy(0, 5);
    
    // Prior for dynamic covariates (NEW)
    to_vector(alpha_dyn) ~ normal(0, 5);
    
    // Prior for measurement covariates
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
        row_vector[p] Xi_row = X[i, ];
        
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + Xi_row * beta[1, ]' + lambda[1] * xi[1, i] + b[sub, 1]);
        if (missing_ID[i, 2] == 0) Y[i, 2] ~ bernoulli_logit(theta2 + Xi_row * beta[2, ]' + lambda[2] * xi[1, i] + b[sub, 2]);
        if (missing_ID[i, 3] == 0) Y[i, 3] ~ bernoulli_logit(theta3 + Xi_row * beta[3, ]' + lambda[3] * xi[1, i] + b[sub, 3]);

        if (missing_ID[i, 4] == 0) Y[i, 4] ~ ordered_logistic(Xi_row * beta[4, ]' + lambda[4] * xi[2, i] + b[sub, 4], theta4);
        if (missing_ID[i, 5] == 0) Y[i, 5] ~ ordered_logistic(Xi_row * beta[5, ]' + lambda[5] * xi[2, i] + b[sub, 5], theta5);
        if (missing_ID[i, 6] == 0) Y[i, 6] ~ ordered_logistic(Xi_row * beta[6, ]' + lambda[6] * xi[2, i] + b[sub, 6], theta6);
        if (missing_ID[i, 7] == 0) Y[i, 7] ~ ordered_logistic(Xi_row * beta[7, ]' + lambda[7] * xi[2, i] + b[sub, 7], theta7);
    }
}