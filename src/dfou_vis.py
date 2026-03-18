import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_all_metrics():
    """Loads all available experiment CSVs and combines them into one DataFrame."""
    files = {
        "ID_Single": "metrics_summary_id_fit.csv",
        "ID_Multi": "metrics_summary_id_multi.csv",
        "Het_Single": "metrics_summary_het_fit.csv",
        "Het_Multi": "metrics_summary_het_multi.csv"
    }
    
    df_list = []
    for exp_name, file_path in files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Experiment'] = exp_name
            df_list.append(df)
        else:
            print(f"Warning: {file_path} not found. Skipping.")
            
    if not df_list:
        raise FileNotFoundError("No metric CSV files found in the current directory.")
        
    return pd.concat(df_list, ignore_index=True)

def plot_metric_with_error_bars(df, metric_prefix, title, ylabel, save_name):
    """
    Creates a grouped bar chart for a specific metric (Mean + Std).
    Groups by Scenario on the X-axis, and uses Experiment + Model_Mode for the bars.
    """
    # Filter to look at the 'dense' data generation (the harder case)
    df_plot = df[df['Data_Mode'] == 'dense'].copy()
    
    # Create a composite category for the legend (e.g., "Het_Multi - dense")
    df_plot['Condition'] = df_plot['Experiment'] + " (" + df_plot['Model_Mode'] + ")"
    
    scenarios = df_plot['Scenario'].unique()
    conditions = df_plot['Condition'].unique()
    
    # Set up the plot geometry
    x = np.arange(len(scenarios))
    width = 0.8 / len(conditions)  
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    mean_col = f"{metric_prefix}_Mean"
    std_col = f"{metric_prefix}_Std"
    
    for i, condition in enumerate(conditions):
        condition_data = df_plot[df_plot['Condition'] == condition]
        
        # Ensure data aligns with the scenario order
        condition_data = condition_data.set_index('Scenario').reindex(scenarios)
        
        means = condition_data[mean_col].values
        stds = condition_data[std_col].values
        
        # Calculate offset for grouped bars
        offset = (i - len(conditions)/2 + 0.5) * width
        
        ax.bar(x + offset, means, width, yerr=stds, label=condition, 
               capsize=4, alpha=0.8, edgecolor='black')

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend(title='Experiment Setup (Model Mode)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_name}")
    plt.close()

if __name__ == "__main__":
    print("Loading metrics data...")
    try:
        df_all = load_all_metrics()
        
        print("\nGenerating comparative plots...")
        # 1. Plot F Trajectory Mean Squared Error
        plot_metric_with_error_bars(
            df=df_all, 
            metric_prefix="F_MSE", 
            title="Latent Trajectory (F) Estimation Error\n(True Data Mode: Dense)", 
            ylabel="Mean Squared Error", 
            save_name="plot_f_mse_comparison.png"
        )
        
        # 2. Plot Theta (Dynamics) Mean Squared Error
        plot_metric_with_error_bars(
            df=df_all, 
            metric_prefix="Theta_MSE", 
            title="Transition Matrix (Theta) Estimation Error\n(True Data Mode: Dense)", 
            ylabel="Mean Squared Error", 
            save_name="plot_theta_mse_comparison.png"
        )
        
        # 3. Plot Factor Loadings (Lambda) Correlation
        plot_metric_with_error_bars(
            df=df_all, 
            metric_prefix="L_Corr", 
            title="Observation Matrix (Lambda) Correlation\n(True Data Mode: Dense)", 
            ylabel="Pearson Correlation (Higher is Better)", 
            save_name="plot_lambda_corr_comparison.png"
        )
        
        # 4. Plot Final Loss Convergence
        plot_metric_with_error_bars(
            df=df_all, 
            metric_prefix="Final_Loss", 
            title="Final EM Loss\n(True Data Mode: Dense)", 
            ylabel="Negative Expected Complete Log-Posterior", 
            save_name="plot_final_loss_comparison.png"
        )

        print("\nAll plots generated successfully!")
        
    except Exception as e:
        print(f"Error executing plotting script: {e}")