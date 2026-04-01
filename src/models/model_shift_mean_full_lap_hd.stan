// Model 2a: Longitudinal IRT with SDE Latent Process (K=12, R=4)

data {
    int<lower=1> N;
    int<lower=1> Nsub;

    int<lower=1> K;                   // Number of items (12)
    int<lower=1> R;                   // Number of latent dimensions (4)
    int<lower=1> p;                   // Item covariates

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

    // Number of categories for ordinal items (Items 6 through 12)
    int<lower=2> ncate6;
    int<lower=2> ncate7;
    int<lower=2> ncate8;
    int<lower=2> ncate9;
    int<lower=2> ncate10;
    int<lower=2> ncate11;
    int<lower=2> ncate12;
}

parameters {
    // Thresholds for Binary Items (1-5)
    real theta1; 
    real theta2; 
    real theta3;
    real theta4; 
    real theta5;

    // Thresholds for Ordinal Items (6-12)
    ordered[ncate6 - 1] theta6;
    ordered[ncate7 - 1] theta7;
    ordered[ncate8 - 1] theta8;
    ordered[ncate9 - 1] theta9;
    ordered[ncate10 - 1] theta10;
    ordered[ncate11 - 1] theta11;
    ordered[ncate12 - 1] theta12;

    real mu_theta;
    real<lower=1e-6> sigma_theta;

    vector<lower=1e-6>[K] lambda;
    real<lower=1e-6> sigma_lambda;

    matrix[K, p] beta;

    matrix[R, q] A_latent;            // slope on covariates * time
    vector[R] c_latent;               // global intercept

    matrix[Nsub, K] b_raw;
    vector<lower=1e-6>[K] sigma_bk;

    array[N] vector[R] xi;

    cholesky_factor_cov[R] L_S;       // SPD component
    
    // 6 elements for the strictly upper triangle of a 4x4 matrix
    vector[6] gamma_skew;             

    // Native Correlation Matrix for Identifiability and HMC Stability
    corr_matrix[R] Omega; 
}

transformed parameters {
    matrix[R,R] S;
    matrix[R,R] A;
    matrix[R,R] Gamma;
    matrix[Nsub, K] b;
    cov_matrix[R] Sigma;

    // SPD component
    S = L_S * L_S';

    // Skew symmetric component mapped from the 6 parameters
    A = rep_matrix(0, R, R);
    A[1, 2] =  gamma_skew[1]; A[2, 1] = -gamma_skew[1];
    A[1, 3] =  gamma_skew[2]; A[3, 1] = -gamma_skew[2];
    A[1, 4] =  gamma_skew[3]; A[4, 1] = -gamma_skew[3];
    A[2, 3] =  gamma_skew[4]; A[3, 2] = -gamma_skew[4];
    A[2, 4] =  gamma_skew[5]; A[4, 2] = -gamma_skew[5];
    A[3, 4] =  gamma_skew[6]; A[4, 3] = -gamma_skew[6];

    // Final drift matrix
    Gamma = S + A;

    // Continuous Lyapunov target mapping (Optional reference tracking)
    Sigma = Gamma * Omega + Omega * Gamma';

    for (i in 1 : Nsub){
        for (k in 1 : K){
            b[i, k] = b_raw[i, k] * sigma_bk[k];
        }
    }
}

model {
    // Priors
    theta1 ~ normal(mu_theta, sigma_theta);
    theta2 ~ normal(mu_theta, sigma_theta);
    theta3 ~ normal(mu_theta, sigma_theta);
    theta4 ~ normal(mu_theta, sigma_theta);
    theta5 ~ normal(mu_theta, sigma_theta);

    theta6 ~ normal(mu_theta, sigma_theta);
    theta7 ~ normal(mu_theta, sigma_theta);
    theta8 ~ normal(mu_theta, sigma_theta);
    theta9 ~ normal(mu_theta, sigma_theta);
    theta10 ~ normal(mu_theta, sigma_theta);
    theta11 ~ normal(mu_theta, sigma_theta);
    theta12 ~ normal(mu_theta, sigma_theta);

    mu_theta ~ normal(0, 10);
    sigma_theta ~ cauchy(0, 5);

    lambda ~ normal(1, sigma_lambda);
    sigma_lambda ~ cauchy(0, 5);

    to_vector(beta) ~ cauchy(0, 5);

    to_vector(A_latent) ~ normal(0, 2);
    c_latent ~ normal(0, 5);

    sigma_bk ~ cauchy(0, 5);
    to_vector(b_raw) ~ normal(0, 1);

    to_vector(L_S) ~ normal(0, 2);
    gamma_skew ~ normal(0, 2);
    
    // Uniform prior over all valid correlation matrices
    Omega ~ lkj_corr(1.0); 

    // Latent Factor Dynamics
    for (i in 1:Nsub) {
        int start_idx = cumu[i] - repme[i] + 1;
        
        // Time = 1
        vector[q] z0 = Z[start_idx]';
        vector[R] mu_start = (A_latent * z0 + c_latent) * time[start_idx];

        xi[start_idx] ~ multi_normal(mu_start, Omega);

        // Time = 2 to end
        for (j in 2:repme[i]) {

            int k = start_idx + j - 1;

            matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
            
            // Stable Covariance Matrix Calculation
            matrix[R, R] Q = Omega - Phi * Omega * Phi';
            matrix[R, R] Q_stable = 0.5 * (Q + Q'); // Ensure symmetry
            
            // xi[k] ~ N(Phi * xi[k-1], Q)
            vector[q] zk = Z[k]';
            vector[q] zk_prev = Z[k-1]';

            vector[R] target_k =
                (A_latent * zk + c_latent) * time[k];

            vector[R] target_prev =
                (A_latent * zk_prev + c_latent) * time[k-1];
                
            vector[R] cond_mean = target_k + Phi * (xi[k-1] - target_prev);
            
            xi[k] ~ multi_normal(cond_mean, add_diag(Q_stable, 1e-6));
        }
    }

    // Likelihood
    for (i in 1:N) {
        int sub = ID[i];
        row_vector[p] Xi_row = X[i, ];
        
        // Factor 1 items (1-3) -- All Binary
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + Xi_row * beta[1, ]' + lambda[1] * xi[i, 1] + b[sub, 1]);
        if (missing_ID[i, 2] == 0) Y[i, 2] ~ bernoulli_logit(theta2 + Xi_row * beta[2, ]' + lambda[2] * xi[i, 1] + b[sub, 2]);
        if (missing_ID[i, 3] == 0) Y[i, 3] ~ bernoulli_logit(theta3 + Xi_row * beta[3, ]' + lambda[3] * xi[i, 1] + b[sub, 3]);

        // Factor 2 items (4-6) -- Items 4,5 Binary; Item 6 Ordinal
        if (missing_ID[i, 4] == 0) Y[i, 4] ~ bernoulli_logit(theta4 + Xi_row * beta[4, ]' + lambda[4] * xi[i, 2] + b[sub, 4]);
        if (missing_ID[i, 5] == 0) Y[i, 5] ~ bernoulli_logit(theta5 + Xi_row * beta[5, ]' + lambda[5] * xi[i, 2] + b[sub, 5]);
        if (missing_ID[i, 6] == 0) Y[i, 6] ~ ordered_logistic(Xi_row * beta[6, ]' + lambda[6] * xi[i, 2] + b[sub, 6], theta6);

        // Factor 3 items (7-9) -- All Ordinal
        if (missing_ID[i, 7] == 0) Y[i, 7] ~ ordered_logistic(Xi_row * beta[7, ]' + lambda[7] * xi[i, 3] + b[sub, 7], theta7);
        if (missing_ID[i, 8] == 0) Y[i, 8] ~ ordered_logistic(Xi_row * beta[8, ]' + lambda[8] * xi[i, 3] + b[sub, 8], theta8);
        if (missing_ID[i, 9] == 0) Y[i, 9] ~ ordered_logistic(Xi_row * beta[9, ]' + lambda[9] * xi[i, 3] + b[sub, 9], theta9);

        // Factor 4 items (10-12) -- All Ordinal
        if (missing_ID[i, 10] == 0) Y[i, 10] ~ ordered_logistic(Xi_row * beta[10, ]' + lambda[10] * xi[i, 4] + b[sub, 10], theta10);
        if (missing_ID[i, 11] == 0) Y[i, 11] ~ ordered_logistic(Xi_row * beta[11, ]' + lambda[11] * xi[i, 4] + b[sub, 11], theta11);
        if (missing_ID[i, 12] == 0) Y[i, 12] ~ ordered_logistic(Xi_row * beta[12, ]' + lambda[12] * xi[i, 4] + b[sub, 12], theta12);
    }
}

generated quantities {
    // Extract the 6 upper-triangular correlations directly for Python evaluation scripts
    vector[6] rho;
    rho[1] = Omega[1, 2];
    rho[2] = Omega[1, 3];
    rho[3] = Omega[1, 4];
    rho[4] = Omega[2, 3];
    rho[5] = Omega[2, 4];
    rho[6] = Omega[3, 4];

    // Stability checks for different time intervals
    matrix[R, R] A05 = matrix_exp(-0.5 * Gamma);
    matrix[R, R] Cova_trans05 = Omega - A05 * Omega * A05';

    matrix[R, R] A10 = matrix_exp(-Gamma);
    matrix[R, R] Cova_trans10 = Omega - A10 * Omega * A10';

    matrix[R, R] A15 = matrix_exp(-1.5 * Gamma);
    matrix[R, R] Cova_trans15 = Omega - A15 * Omega * A15';
}