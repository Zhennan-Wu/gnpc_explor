import torch
import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
import warnings

# Suppress scipy warnings for very small sample sizes in Shapiro-Wilk if needed
warnings.filterwarnings("ignore", category=UserWarning)

def get_stars(p):
    """Formats p-values into significance markers."""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def compare_distributions(dist_a, dist_b):
    """
    Dynamically chooses between Welch's t-test and Mann-Whitney U test
    based on the Shapiro-Wilk test for normality.
    """
    # 1. Test for normality (Shapiro-Wilk)
    # H0: Data is drawn from a normal distribution.
    _, shapiro_p_a = stats.shapiro(dist_a)
    _, shapiro_p_b = stats.shapiro(dist_b)
    
    alpha_normality = 0.05
    is_normal = (shapiro_p_a >= alpha_normality) and (shapiro_p_b >= alpha_normality)
    
    # 2. Choose and execute the appropriate test
    if is_normal:
        test_used = "Welch's t-test"
        stat, p_val = stats.ttest_ind(dist_a, dist_b, equal_var=False)
    else:
        test_used = "Mann-Whitney U"
        stat, p_val = stats.mannwhitneyu(dist_a, dist_b, alternative='two-sided')
        
    return p_val, test_used

def calculate_statistical_significance(archive_path):
    """
    Loads a parameter archive, extracts run-level metrics, and performs 
    dynamic statistical testing to compare 'diagonal' vs 'dense' model performance.
    """
    if not os.path.exists(archive_path):
        return None
        
    print(f"Loading archive: {archive_path}")
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    
    # Parse unique scenarios and data generation modes
    scenarios = set()
    data_modes = set()
    for key in archive.keys():
        parts = key.split("_")
        scenarios.add(parts[0])
        data_modes.add(parts[1].split("-")[1])
        
    scenarios = sorted(list(scenarios))
    data_modes = sorted(list(data_modes))
    
    results = []
    
    for scenario in scenarios:
        for d_mode in data_modes:
            f_mse_diag, f_mse_dense = [], []
            th_mse_diag, th_mse_dense = [], []
            
            # Extract metrics for all runs
            for run_idx in range(10):  
                key_diag = f"{scenario}_Data-{d_mode}_Model-diagonal_Run-{run_idx}"
                key_dense = f"{scenario}_Data-{d_mode}_Model-dense_Run-{run_idx}"
                
                if key_diag in archive:
                    f_mse_diag.append(archive[key_diag]['metrics']['F_mse'])
                    th_mse_diag.append(archive[key_diag]['metrics']['Theta_mse'])
                
                if key_dense in archive:
                    f_mse_dense.append(archive[key_dense]['metrics']['F_mse'])
                    th_mse_dense.append(archive[key_dense]['metrics']['Theta_mse'])
            
            # Require at least 3 samples to perform meaningful normality/significance tests
            if len(f_mse_diag) >= 3 and len(f_mse_dense) >= 3:
                
                # Dynamic Testing for F_MSE
                p_val_f, test_f = compare_distributions(f_mse_diag, f_mse_dense)
                f_winner = "Dense" if np.mean(f_mse_dense) < np.mean(f_mse_diag) else "Diagonal"
                
                # Dynamic Testing for Theta_MSE
                p_val_th, test_th = compare_distributions(th_mse_diag, th_mse_dense)
                th_winner = "Dense" if np.mean(th_mse_dense) < np.mean(th_mse_diag) else "Diagonal"
                
                results.append({
                    "Scenario": scenario,
                    "True_Data_Mode": d_mode,
                    "F_MSE_Diag_Mean": np.mean(f_mse_diag),
                    "F_MSE_Dense_Mean": np.mean(f_mse_dense),
                    "F_Better_Model": f_winner,
                    "F_Test_Used": test_f,
                    "F_P_Value": p_val_f,
                    "F_Sig": get_stars(p_val_f),
                    "Theta_MSE_Diag_Mean": np.mean(th_mse_diag),
                    "Theta_MSE_Dense_Mean": np.mean(th_mse_dense),
                    "Theta_Better_Model": th_winner,
                    "Theta_Test_Used": test_th,
                    "Theta_P_Value": p_val_th,
                    "Theta_Sig": get_stars(p_val_th)
                })
                
    return pd.DataFrame(results)

if __name__ == "__main__":
    archive_files = glob.glob("parameter_archive_*.pt")
    
    if not archive_files:
        print("No .pt archive files found in the current directory.")
    else:
        for file in archive_files:
            df_stats = calculate_statistical_significance(file)
            
            if df_stats is not None and not df_stats.empty:
                print(f"\n{'='*120}")
                print(f" STATISTICAL SIGNIFICANCE REPORT: {file}")
                print(f"{'='*120}")
                
                # Display a clean console version focusing on the dense data generation (the hardest test)
                display_df = df_stats[df_stats['True_Data_Mode'] == 'dense'].copy()
                display_df = display_df[['Scenario', 'F_Better_Model', 'F_Test_Used', 'F_Sig', 
                                         'Theta_Better_Model', 'Theta_Test_Used', 'Theta_Sig']]
                
                print("\n[ Filtering display for True_Data_Mode = 'dense' ]")
                print(display_df.to_string(index=False))
                
                # Save the full detailed stats to CSV
                out_name = file.replace("parameter_archive_", "statistical_tests_").replace(".pt", ".csv")
                df_stats.to_csv(out_name, index=False)
                print(f"\n[+] Saved detailed statistical report to '{out_name}'")