import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set a clean, modern aesthetic for the plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_em_convergence(archive_path="parameter_archive_het_multi.pt"):
    """
    Loads a parameter archive and plots the EM loss convergence history.
    Saves a separate plot for each scenario comparing the model modes.
    """
    if not os.path.exists(archive_path):
        print(f"[-] Error: Could not find '{archive_path}'. Please check the path.")
        return
        
    print(f"[+] Loading parameter archive: {archive_path}")
    # weights_only=False is required to load standard Python dictionaries containing tensors
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    
    # Extract unique scenario names dynamically from the keys
    # Key format: "{scenario_name}_Data-{d_mode}_Model-{m_mode}_Run-{run_idx}"
    scenarios = set()
    for key in archive.keys():
        scenario_name = key.split("_Data-")[0]
        scenarios.add(scenario_name)
        
    scenarios = sorted(list(scenarios))
    
    # We focus on the "dense" data generation as the primary stress test
    data_mode = "dense"
    model_modes = ["diagonal", "dense"]
    
    # Color palette for distinction
    colors = {"diagonal": "#d62728", "dense": "#1f77b4"} 
    
    output_dir = "convergence_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Generating plots in '{output_dir}/'...\n")
    
    for scenario in scenarios:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for m_mode in model_modes:
            histories = []
            
            # Iterate through runs (assuming up to 10 runs based on the previous scripts)
            for run_idx in range(10):
                key = f"{scenario}_Data-{data_mode}_Model-{m_mode}_Run-{run_idx}"
                if key in archive:
                    # Move to CPU/numpy if it happens to be a tensor, though it's likely a standard list
                    history = archive[key]['loss_history']
                    histories.append(history)
            
            if not histories:
                continue
                
            # Truncate to the minimum length in case multistart burn-in epochs varied the array size
            min_len = min(len(h) for h in histories)
            histories_np = np.array([h[:min_len] for h in histories])
            
            epochs = np.arange(1, min_len + 1)
            mean_loss = np.mean(histories_np, axis=0)
            
            # 1. Plot individual runs with low opacity (spaghetti plot)
            for h in histories_np:
                ax.plot(epochs, h, color=colors[m_mode], alpha=0.15, linewidth=1.5)
            
            # 2. Plot the mean trend line on top
            ax.plot(epochs, mean_loss, color=colors[m_mode], alpha=1.0, linewidth=3, 
                     label=f"Model: {m_mode.capitalize()} (Mean of {len(histories)} runs)")

        ax.set_title(f"EM Convergence: {scenario}\n(True Data Mode: {data_mode.capitalize()})", 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Main EM Iteration", fontsize=12, fontweight='bold')
        ax.set_ylabel("Negative Expected Complete Log-Posterior", fontsize=12, fontweight='bold')
        
        ax.legend(loc="upper right", frameon=True, shadow=True, fancybox=True)
        
        plt.tight_layout()
        
        # Format a clean filename
        safe_name = scenario.replace(" ", "_").replace(".", "").lower()
        save_path = os.path.join(output_dir, f"convergence_{safe_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  -> Saved: {save_path}")

if __name__ == "__main__":
    # Point this to whichever archive you just finished running
    target_archive = "parameter_archive_het_multi.pt"
    plot_em_convergence(target_archive)