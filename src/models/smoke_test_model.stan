
data {
    int N; int Nsub; int K; int R; int p; int q;
    array[N] int ID; array[Nsub] int cumu; array[Nsub] int repme;
    array[N, K] int Y; array[N, K] int missing_ID;
    vector[N] deltat; vector[N] time;
    matrix[N, p] X; matrix[N, q] Z;
    int ncate4; int ncate5; int ncate6; int ncate7;
}
parameters {
    real theta1; real theta2; real theta3;
    ordered[ncate4 - 1] theta4; ordered[ncate5 - 1] theta5; 
    ordered[ncate6 - 1] theta6; ordered[ncate7 - 1] theta7;
    vector<lower=1e-6>[K] lambda; matrix[K, p] beta;
    matrix[R, q] A_latent; matrix[R, q] B_latent; vector[R] c_latent;
    matrix[R, R] Gamma; real<lower=-1, upper=1> rho;
}
model {
    theta1 ~ std_normal(); theta2 ~ std_normal(); theta3 ~ std_normal();
    lambda ~ lognormal(0, 1); to_vector(beta) ~ std_normal();
    to_vector(A_latent) ~ std_normal(); to_vector(B_latent) ~ std_normal();
    c_latent ~ std_normal(); to_vector(Gamma) ~ std_normal(); rho ~ uniform(-1, 1);
}
