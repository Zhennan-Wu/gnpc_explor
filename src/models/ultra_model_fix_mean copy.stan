functions {
  // Helper to solve the Continuous Lyapunov Equation: Gamma * Omega + Omega * Gamma' = Sigma
  // Solves (I \otimes Gamma + Gamma \otimes I) vec(Omega) = vec(Sigma)
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
    
    // Vectorize Sigma
    for (i in 1:R) {
      for (j in 1:R) {
        vec_Sigma[(j - 1) * R + i] = Sigma[i, j];
      }
    }
    
    // Solve linear system
    vec_Omega = K \ vec_Sigma;
    
    // Reshape back to matrix
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

    vector<lower=1e-6>[K] lambda;
    real<lower=1e-6> sigma_lambda;

    matrix[K, p] beta;

    matrix[Nsub, K] b_raw;
    vector<lower=1e-6>[K] sigma_bk;

    // --- Structural Parameters (Corrected) ---
    cholesky_factor_corr[R] L_S_corr;     // Correlation base for S
    vector<lower=0>[R] L_S_scale;         // Scales for S
    
    vector[R * (R - 1) / 2] a_low;        // Skew-symmetric base for Drift
    
    cholesky_factor_corr[R] L_Sigma_corr; // Correlation base for Sigma
    vector<lower=0>[R] L_Sigma_scale;     // Scales for Sigma
    
    // --- Latent States (Non-Centered GMRF) ---
    matrix[R, N] xi_raw;
}

transformed parameters {
    matrix[Nsub, K] b;
    
    // Structural Matrices
    matrix[R, R] S;
    matrix[R, R] A;
    matrix[R, R] Gamma;
    matrix[R, R] Sigma;
    matrix[R, R] Omega;
    
    // The actual latent states mapped to the observations
    matrix[R, N] xi;

    // 1. Calculate random effects
    for (i in 1:Nsub){
        for (k in 1:K){
            b[i, k] = b_raw[i, k] * sigma_bk[k];
        }
    }
    
    // 2. Construct stable Gamma = S + A
    matrix[R, R] L_S = diag_pre_multiply(L_S_scale, L_S_corr);
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
    
    // 3. Construct diffusion matrix and solve Lyapunov for Omega
    matrix[R, R] L_Sigma = diag_pre_multiply(L_Sigma_scale, L_Sigma_corr);
    Sigma = multiply_lower_tri_self_transpose(L_Sigma);
    
    Omega = solve_lyapunov(Gamma, Sigma, R);
    Omega = 0.5 * (Omega + Omega'); // Explicitly symmetrize to fix floating-point drift
    Omega = add_diag(Omega, 1e-5);  // Add tiny nugget for absolute positive-definite safety
    
    // 4. Generate Latent Trajectories (Joint Precision Equivalency)
    {
        matrix[R, R] L_Omega = cholesky_decompose(Omega);
        
        for (i in 1:Nsub) {
            int start_idx = cumu[i] - repme[i] + 1;
            
            // Time 1: Drawn from stationary distribution
            xi[:, start_idx] = L_Omega * xi_raw[:, start_idx];
            
            // Time 2 to end: Non-centered conditional draws
            for (j in 2:repme[i]) {
                int k = start_idx + j - 1;
                matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
                
                // Conditional Covariance Q = Omega - Phi * Omega * Phi'
                matrix[R, R] Q = Omega - Phi * Omega * Phi';
                matrix[R, R] Q_sym = 0.5 * (Q + Q'); // Symmetrize for numeric stability
                matrix[R, R] L_Q = cholesky_decompose(add_diag(Q_sym, 1e-6));
                
                // Deterministic mapping equivalent to the sequential draw
                xi[:, k] = Phi * xi[:, k-1] + L_Q * xi_raw[:, k];
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
    lambda ~ normal(1, sigma_lambda);
    sigma_lambda ~ cauchy(0, 5);
    to_vector(beta) ~ cauchy(0, 5);
    sigma_bk ~ cauchy(0, 5);
    to_vector(b_raw) ~ normal(0, 1);

    // Priors (Structural & Latent)
    L_S_corr ~ lkj_corr_cholesky(2.0);
    L_S_scale ~ lognormal(0, 0.5); // Tight prior to prevent exploding drift
    
    a_low ~ normal(0, 0.5);
    
    L_Sigma_corr ~ lkj_corr_cholesky(2.0);
    L_Sigma_scale ~ lognormal(0, 0.5);
    
    to_vector(xi_raw) ~ std_normal();

    // --- Likelihood (IRT) ---
    for (i in 1:N) {
        int sub = ID[i];
        row_vector[p] Xi_row = X[i, ];
        
        // Factor 1 items (1-3)
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + Xi_row * beta[1, ]' + lambda[1] * xi[1, i] + b[sub, 1]);
        if (missing_ID[i, 2] == 0) Y[i, 2] ~ bernoulli_logit(theta2 + Xi_row * beta[2, ]' + lambda[2] * xi[1, i] + b[sub, 2]);
        if (missing_ID[i, 3] == 0) Y[i, 3] ~ bernoulli_logit(theta3 + Xi_row * beta[3, ]' + lambda[3] * xi[1, i] + b[sub, 3]);

        // Factor 2 items (4-7)
        if (missing_ID[i, 4] == 0) Y[i, 4] ~ ordered_logistic(Xi_row * beta[4, ]' + lambda[4] * xi[2, i] + b[sub, 4], theta4);
        if (missing_ID[i, 5] == 0) Y[i, 5] ~ ordered_logistic(Xi_row * beta[5, ]' + lambda[5] * xi[2, i] + b[sub, 5], theta5);
        if (missing_ID[i, 6] == 0) Y[i, 6] ~ ordered_logistic(Xi_row * beta[6, ]' + lambda[6] * xi[2, i] + b[sub, 6], theta6);
        if (missing_ID[i, 7] == 0) Y[i, 7] ~ ordered_logistic(Xi_row * beta[7, ]' + lambda[7] * xi[2, i] + b[sub, 7], theta7);
    }
}