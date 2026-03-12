// Model 2a: Continuous-Time Dynamic Factor Analysis
data {
    int<lower=1> N;                   // Total number of observations
    int<lower=1> Nsub;                // Number of subjects
    int<lower=1> K;                   // Number of items
    int<lower=1> R;                   // Number of latent factors
    int<lower=1> p;                   // Number of covariates

    array[N] int<lower=1, upper=Nsub> ID;
    array[Nsub] int cumu;             // Cumulative indices
    array[Nsub] int repme;            // Observations per subject

    array[N, K] int Y;                // Response matrix
    array[N, K] int missing_ID;       // 0 = present, 1 = missing

    vector[N] deltat;
    matrix[N, p] X;

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
    real<lower=0> sigma_theta;

    vector<lower=0>[K] lambda;
    real<lower=0> sigma_lambda;

    matrix[K, p] beta;

    matrix[Nsub, K] b_raw;
    vector<lower=0>[K] sigma_bk;

    array[N] vector[R] xi;           // Latent factors

    matrix[R, R] Gamma;              // Drift matrix
    real<lower=-1, upper=1> rho;     // Factor correlation
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
    to_vector(Gamma) ~ normal(0, 10);

    // Latent Factor Dynamics (Continuous Time)
    for (i in 1:Nsub) {
        int start_idx = cumu[i] - repme[i] + 1;
        
        // Initial state
        xi[start_idx] ~ multi_normal([0, 0]', Omega);

        // Transitions
        for (j in 2:repme[i]) {
            int k = start_idx + j - 1;
            matrix[R, R] Phi = matrix_exp(-deltat[k] * Gamma);
            matrix[R, R] Q = Omega - Phi * Omega * Phi';
            
            // Ensure Q is positive definite for sampling
            xi[k] ~ multi_normal(Phi * xi[k-1], Q);
        }
    }

    // Likelihood
    for (i in 1:N) {
        // Binary items 1-3
        if (missing_ID[i, 1] == 0) Y[i, 1] ~ bernoulli_logit(theta1 + X[i, ] * beta[1, ]' + lambda[1] * xi[i, 1] + b[ID[i], 1]);
        if (missing_ID[i, 2] == 0) Y[i, 2] ~ bernoulli_logit(theta2 + X[i, ] * beta[2, ]' + lambda[2] * xi[i, 1] + b[ID[i], 2]);
        if (missing_ID[i, 3] == 0) Y[i, 3] ~ bernoulli_logit(theta3 + X[i, ] * beta[3, ]' + lambda[3] * xi[i, 1] + b[ID[i], 3]);

        // Ordinal items 4-7
        if (missing_ID[i, 4] == 0) Y[i, 4] ~ ordered_logistic(X[i, ] * beta[4, ]' + lambda[4] * xi[i, 2] + b[ID[i], 4], theta4);
        if (missing_ID[i, 5] == 0) Y[i, 5] ~ ordered_logistic(X[i, ] * beta[5, ]' + lambda[5] * xi[i, 2] + b[ID[i], 5], theta5);
        if (missing_ID[i, 6] == 0) Y[i, 6] ~ ordered_logistic(X[i, ] * beta[6, ]' + lambda[6] * xi[i, 2] + b[ID[i], 6], theta6);
        if (missing_ID[i, 7] == 0) Y[i, 7] ~ ordered_logistic(X[i, ] * beta[7, ]' + lambda[7] * xi[i, 2] + b[ID[i], 7], theta7);
    }
}
