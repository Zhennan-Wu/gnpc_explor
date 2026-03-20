
data {
    int N; int Nsub; int K; int R;
    array[N] int ID; array[Nsub] int cumu; array[Nsub] int repme;
    array[N, K] int Y; vector[N] deltat; int ncate;
}
parameters {
    real theta1; real theta2; real theta3;
    ordered[ncate - 1] theta4; ordered[ncate - 1] theta5; 
    ordered[ncate - 1] theta6; ordered[ncate - 1] theta7;
    vector<lower=1e-6>[K - 2] lambda_free;
    matrix[Nsub, K] b_raw; vector<lower=1e-6>[K] sigma_bk;
    
    // Direct Parameterization of Gamma with Hard Constraints
    matrix[R, R] Gamma;
    cholesky_factor_corr[R] L_Sigma_corr; vector<lower=0>[R] L_Sigma_scale;
    
    // Centered Latent State (The Funnel Trap)
    matrix[R, N] xi; 
}
transformed parameters {
    matrix[Nsub, K] b; vector[K] lambda; matrix[R, R] Sigma;
    
    // Routh-Hurwitz Stability Constraints for R=2
    real<lower=0.0001> constraint1 = Gamma[1,1] + Gamma[2,2];
    real<lower=0.0001> constraint2 = Gamma[1,1]*Gamma[2,2] - Gamma[1,2]*Gamma[2,1];

    lambda[1] = 1.0; lambda[2] = lambda_free[1]; lambda[3] = lambda_free[2];
    lambda[4] = 1.0; lambda[5] = lambda_free[3]; lambda[6] = lambda_free[4]; lambda[7] = lambda_free[5];
    for (i in 1:Nsub) { for (k in 1:K) { b[i, k] = b_raw[i, k] * sigma_bk[k]; } }
    Sigma = multiply_lower_tri_self_transpose(diag_pre_multiply(L_Sigma_scale, L_Sigma_corr));
}
model {
    lambda_free ~ normal(1, 2); sigma_bk ~ cauchy(0, 2); to_vector(b_raw) ~ std_normal();
    to_vector(Gamma) ~ normal(0, 2); 
    L_Sigma_corr ~ lkj_corr_cholesky(2.0); L_Sigma_scale ~ lognormal(0, 0.5);

    // Solve Lyapunov for Time 1 stationary distribution (Vectorized manual approach for baseline)
    matrix[R*R, R*R] K_lyap; vector[R*R] vec_Sigma; vector[R*R] vec_Omega; matrix[R, R] Omega;
    for (i in 1:R) { for (j in 1:R) { for (k in 1:R) { for (l in 1:R) {
            int row_idx = (j - 1) * R + i; int col_idx = (l - 1) * R + k;
            K_lyap[row_idx, col_idx] = (i == k ? Gamma[j, l] : 0.0) + (j == l ? Gamma[i, k] : 0.0);
    } } } }
    for (i in 1:R) { for (j in 1:R) { vec_Sigma[(j - 1) * R + i] = Sigma[i, j]; } }
    vec_Omega = K_lyap \ vec_Sigma;
    for (i in 1:R) { for (j in 1:R) { Omega[i, j] = vec_Omega[(j - 1) * R + i]; } }
    Omega = 0.5 * (Omega + Omega'); Omega = add_diag(Omega, 1e-5);

    // SEQUENTIAL SAMPLING
    for (i in 1:Nsub) {
        int start_idx = cumu[i] - repme[i] + 1;
        xi[:, start_idx] ~ multi_normal(rep_vector(0, R), Omega); // Time 1
        
        for (j in 2:repme[i]) {
            int k = start_idx + j - 1;
            matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
            matrix[R, R] Q = Omega - Phi * Omega * Phi';
            matrix[R, R] Q_sym = add_diag(0.5 * (Q + Q'), 1e-6);
            
            // The Centered Bottleneck
            xi[:, k] ~ multi_normal(Phi * xi[:, k-1], Q_sym); 
        }
    }

    for (i in 1:N) {
        int sub = ID[i];
        Y[i, 1] ~ bernoulli_logit(theta1 + lambda[1] * xi[1, i] + b[sub, 1]);
        Y[i, 2] ~ bernoulli_logit(theta2 + lambda[2] * xi[1, i] + b[sub, 2]);
        Y[i, 3] ~ bernoulli_logit(theta3 + lambda[3] * xi[1, i] + b[sub, 3]);
        Y[i, 4] ~ ordered_logistic(lambda[4] * xi[2, i] + b[sub, 4], theta4);
        Y[i, 5] ~ ordered_logistic(lambda[5] * xi[2, i] + b[sub, 5], theta5);
        Y[i, 6] ~ ordered_logistic(lambda[6] * xi[2, i] + b[sub, 6], theta6);
        Y[i, 7] ~ ordered_logistic(lambda[7] * xi[2, i] + b[sub, 7], theta7);
    }
}
