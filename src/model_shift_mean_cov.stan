// Model 2a: Longitudinal IRT with SDE Latent Process

data {
    int<lower=1> N;
    int<lower=1> Nsub;

    int<lower=1> K;
    int<lower=1> R;
    int<lower=1> p;

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

    matrix[R, q] A_latent;     // slope on covariates * time

    matrix[Nsub, K] b_raw;
    vector<lower=1e-6>[K] sigma_bk;

    array[N] vector[R] xi;

    matrix[R, R] Gamma;

    real<lower=-1, upper=1> rho; 
}

transformed parameters {

	matrix[Nsub, K] b;

	real<lower=0.000001> constraint1;
	real<lower=0.000001> constraint2;

	corr_matrix[R] Omega;
	cov_matrix[R] Sigma;

	constraint1 = Gamma[1, 1] + Gamma[2, 2];
	constraint2 = Gamma[1, 1] * Gamma[2, 2] - Gamma[1, 2] * Gamma[2, 1];

	Omega = [[1, rho], [rho, 1]];
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

    mu_theta ~ normal(0, 10);
    sigma_theta ~ cauchy(0, 5);

    lambda ~ normal(1, sigma_lambda);
    sigma_lambda ~ cauchy(0, 5);

    to_vector(beta) ~ cauchy(0, 5);

    to_vector(A_latent) ~ normal(0, 2);
    c_latent ~ normal(0, 5);

    sigma_bk ~ cauchy(0, 5);
    to_vector(b_raw) ~ normal(0, 1);

    to_vector(Gamma) ~ normal(0, 10);

    // Latent Factor Dynamics
    for (i in 1:Nsub) {
        int start_idx = cumu[i] - repme[i] + 1;
        
        // Time = 1
        vector[q] z0 = Z[start_idx]';
        vector[R] mu_start = (A_latent * z0) * time[start_idx];

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
                (A_latent * zk) * time[k];

            vector[R] target_prev =
                (A_latent * zk_prev) * time[k-1];
                
            vector[R] cond_mean = target_k + Phi * (xi[k-1] - target_prev);
            
            xi[k] ~ multi_normal(cond_mean, add_diag(Q_stable, 1e-6));
        }
    }

    // Likelihood
    for (i in 1:N) {
        int sub = ID[i];
        row_vector[p] Xi_row = X[i, ];
        
        // Factor 1 items (1-3)
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + Xi_row * beta[1, ]' + lambda[1] * xi[i, 1] + b[sub, 1]);
        if (missing_ID[i, 2] == 0) Y[i, 2] ~ bernoulli_logit(theta2 + Xi_row * beta[2, ]' + lambda[2] * xi[i, 1] + b[sub, 2]);
        if (missing_ID[i, 3] == 0) Y[i, 3] ~ bernoulli_logit(theta3 + Xi_row * beta[3, ]' + lambda[3] * xi[i, 1] + b[sub, 3]);

        // Factor 2 items (4-7)
        if (missing_ID[i, 4] == 0) Y[i, 4] ~ ordered_logistic(Xi_row * beta[4, ]' + lambda[4] * xi[i, 2] + b[sub, 4], theta4);
        if (missing_ID[i, 5] == 0) Y[i, 5] ~ ordered_logistic(Xi_row * beta[5, ]' + lambda[5] * xi[i, 2] + b[sub, 5], theta5);
        if (missing_ID[i, 6] == 0) Y[i, 6] ~ ordered_logistic(Xi_row * beta[6, ]' + lambda[6] * xi[i, 2] + b[sub, 6], theta6);
        if (missing_ID[i, 7] == 0) Y[i, 7] ~ ordered_logistic(Xi_row * beta[7, ]' + lambda[7] * xi[i, 2] + b[sub, 7], theta7);
    }
}

generated quantities {
    // Stability checks for different time intervals
    matrix[R, R] A05 = matrix_exp(-0.5 * Gamma);
    matrix[R, R] Cova_trans05 = Omega - A05 * Omega * A05';

    matrix[R, R] A10 = matrix_exp(-Gamma);
    matrix[R, R] Cova_trans10 = Omega - A10 * Omega * A10';

    matrix[R, R] A15 = matrix_exp(-1.5 * Gamma);
    matrix[R, R] Cova_trans15 = Omega - A15 * Omega * A15';
}