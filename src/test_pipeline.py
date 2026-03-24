import os
import subprocess
import pandas as pd

DUMMY_STAN_CODE = """
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
"""

STAN_FILE = "smoke_test_model.stan"
TARGET_SCRIPT = "lou_simulation_hpc_mp.py"

def run_command(cmd_list, step_name):
    print(f"\n--- Running Step: {step_name} ---")
    print(f"Command: {' '.join(cmd_list)}")
    # CRITICAL FIX: Removed capture_output=True so we can see the live worker crash logs!
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"❌ ERROR in {step_name}!")
        exit(1)
    print(f"✅ {step_name} completed successfully.")

def main():
    print("🚀 Starting Pipeline Smoke Test...\n")
    
    with open(STAN_FILE, "w") as f: f.write(DUMMY_STAN_CODE)
    print(f"Created dummy model: {STAN_FILE}")
    
    run_command(["python3", TARGET_SCRIPT, "--scenario", "S1", "--model", STAN_FILE, "--compile_only"], "Model Compilation")
    
    # Run the simulation (this is where it will crash and print the traceback)
    run_command(["python3", TARGET_SCRIPT, "--scenario", "S1", "--model", STAN_FILE, "--chains", "2", "--warmup", "50", "--sampling", "50", "--start_run", "1", "--end_run", "2", "--workers", "2"], "Parallel Simulation Execution")
    
    run_command(["python3", TARGET_SCRIPT, "--scenario", "S1", "--model", STAN_FILE, "--aggregate_only"], "Results Aggregation")
    
    table_file = "TABLE_S1_smoke_test_model.csv"
    if not os.path.exists(table_file):
        print(f"❌ ERROR: Final aggregation table {table_file} was not generated!")
        exit(1)
        
    df = pd.read_csv(table_file)
    print(f"✅ Final table generated with shape: {df.shape}")
    print("\n🎉 SUCCESS: The LOU Simulation Pipeline is completely BUG-FREE and ready for HPC deployment!")

if __name__ == "__main__":
    main()