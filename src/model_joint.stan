
functions {
  matrix solve_lyapunov(matrix Gamma, matrix Sigma, int R) {
    matrix[R*R, R*R] K; vector[R*R] vec_Sigma; vector[R*R] vec_Omega; matrix[R, R] Omega;
    for (i in 1:R) { for (j in 1:R) { for (k in 1:R) { for (l in 1:R) {
            int row_idx = (j - 1) * R + i; int col_idx = (l - 1) * R + k;
            K[row_idx, col_idx] = (i == k ? Gamma[j, l] : 0.0) + (j == l ? Gamma[i, k] : 0.0);
    } } } }
    for (i in 1:R) { for (j in 1:R) { vec_Sigma[(j - 1) * R + i] = Sigma[i, j]; } }
    vec_Omega = K \ vec_Sigma;
    for (i in 1:R) { for (j in 1:R) { Omega[i, j] = vec_Omega[(j - 1) * R + i]; } }
    return Omega;
  }
}
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
    
    // S + A Structural parameters
    cholesky_factor_corr[R] L_S_corr; vector<lower=0>[R] L_S_scale;
    vector[R * (R - 1) / 2] a_low;
    cholesky_factor_corr[R] L_Sigma_corr; vector<lower=0>[R] L_Sigma_scale;
    
    // Non-Centered Latent State
    matrix[R, N] xi_raw; 
}
transformed parameters {
    matrix[Nsub, K] b; vector[K] lambda; matrix[R, N] xi;
    matrix[R, R] Gamma; matrix[R, R] Sigma; matrix[R, R] Omega;
    
    lambda[1] = 1.0; lambda[2] = lambda_free[1]; lambda[3] = lambda_free[2];
    lambda[4] = 1.0; lambda[5] = lambda_free[3]; lambda[6] = lambda_free[4]; lambda[7] = lambda_free[5];

    for (i in 1:Nsub) { for (k in 1:K) { b[i, k] = b_raw[i, k] * sigma_bk[k]; } }
    
    Gamma = multiply_lower_tri_self_transpose(diag_pre_multiply(L_S_scale, L_S_corr));
    Gamma[2, 1] = Gamma[2, 1] - a_low[1]; Gamma[1, 2] = Gamma[1, 2] + a_low[1]; // S + A
    
    Sigma = multiply_lower_tri_self_transpose(diag_pre_multiply(L_Sigma_scale, L_Sigma_corr));
    Omega = solve_lyapunov(Gamma, Sigma, R);
    Omega = 0.5 * (Omega + Omega'); Omega = add_diag(Omega, 1e-5);
    
    {
        matrix[R, R] L_Omega = cholesky_decompose(Omega);
        for (i in 1:Nsub) {
            int start_idx = cumu[i] - repme[i] + 1;
            xi[:, start_idx] = L_Omega * xi_raw[:, start_idx];
            for (j in 2:repme[i]) {
                int k = start_idx + j - 1;
                matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
                matrix[R, R] Q = Omega - Phi * Omega * Phi';
                matrix[R, R] L_Q = cholesky_decompose(add_diag(0.5 * (Q + Q'), 1e-6));
                xi[:, k] = Phi * xi[:, k-1] + L_Q * xi_raw[:, k];
            }
        }
    }
}
model {
    lambda_free ~ normal(1, 2); sigma_bk ~ cauchy(0, 2); to_vector(b_raw) ~ std_normal();
    L_S_corr ~ lkj_corr_cholesky(2.0); L_S_scale ~ lognormal(0, 0.5); a_low ~ normal(0, 0.5);
    L_Sigma_corr ~ lkj_corr_cholesky(2.0); L_Sigma_scale ~ lognormal(0, 0.5);
    to_vector(xi_raw) ~ std_normal();

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
