import os
import glob

def generate_experiment_list(output_file="experiments.txt", stan_dir=".", scenarios=None):
    """
    Scans for .stan files and generates a text file with all combinations
    of scenarios and models for a Slurm job array.
    """
    if scenarios is None:
        # Define the data generation scenarios you want to run
        # scenarios = ['S1', 'S2', 'S3', 'S4', 'S5']
        scenarios = ['S2', 'S3', 'S4', 'S5']
        
    # Find all .stan files in the specified directory
    stan_files = glob.glob(os.path.join(stan_dir, "*.stan"))
    
    # Extract just the filenames so the text file is clean
    stan_filenames = [os.path.basename(f) for f in stan_files]
    
    if not stan_filenames:
        print(f"Warning: No .stan files found in the directory: '{stan_dir}'.")
        return

    # Write the combinations to the text file
    with open(output_file, 'w') as f:
        count = 0
        for scenario in scenarios:
            for stan_file in stan_filenames:
                f.write(f"{scenario} {stan_file}\n")
                count += 1
                
    print(f"Successfully generated '{output_file}' with {count} total jobs.")
    print(f"IMPORTANT: Update your Slurm script to use '#SBATCH --array=1-{count}'")

if __name__ == "__main__":
    generate_experiment_list()