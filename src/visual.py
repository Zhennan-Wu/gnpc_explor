import os
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set global publication style
plt.style.use('seaborn-v0_8-paper') 
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": False, # Set to True if you have a local LaTeX distribution
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5
})

class ModelVisualizer:
    def __init__(self, gt_params, gt_data, save_dir="../viz_results"):
        self.save_dir = save_dir
        self.gt = gt_params
        self.gt_data = gt_data
        os.makedirs(self.save_dir, exist_ok=True)
        # Professional color palette
        self.palette = sns.color_palette("colorblind")

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def _compute_alignment(self, model):
        """Calculates the Orthogonal Procrustes rotation matrix R."""
        Le = self._to_numpy(model.get_Lambda())
        Lt = self._to_numpy(self.gt['Lambda'])
        U, _, Vt = scipy.linalg.svd(Lt.T @ Le)
        # Vt.T @ U.T aligns estimated (Le) to ground truth (Lt)
        return torch.tensor(Vt.T @ U.T, device=model.device, dtype=torch.float32)
    
    def _latent_trajectory_estimation(self, model, data, times, covs):
        model.eval()
        with torch.no_grad():
            R = self._compute_alignment(model)
            f_s, _, _ = model.kalman_filter_smoother(data, times, covs)
            traj_aligned = (f_s @ R).cpu().numpy()
        return traj_aligned

    def plot_multi_model_metrics(self, models):
        """Compares convergence metrics across models with shaded error bars."""
        models_results = {name: [run.get_history() for run in runs] for name, runs in models.items()}
        first_model = list(models_results.keys())[0]
        metrics = [k for k in models_results[first_model][0].keys() if models_results[first_model][0][k]]
        
        n_metrics = len(metrics)
        cols = min(3, n_metrics)
        rows = math.ceil(n_metrics / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            for m_idx, (model_name, runs) in enumerate(models_results.items()):
                if metric not in runs[0]:
                    continue
                data = np.array([run[metric] for run in runs])
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                epochs = np.arange(len(mean))

                color = self.palette[m_idx % len(self.palette)]
                ax.plot(epochs, mean, label=model_name, color=color, zorder=3)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            if any(m in metric.lower() for m in ['mse', 'err', 'loss', 'likelihood']):
                ax.set_yscale('log')
            
            sns.despine(ax=ax)
            if i == 0:
                ax.legend(frameon=True, facecolor='white', framealpha=0.8)

        # Remove unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.savefig(os.path.join(self.save_dir, "multi_model_metrics.pdf"), bbox_inches='tight')
        plt.close()

    def plot_multi_model_trajectories(self, models):
        """Plots latent factor recovery vs Ground Truth."""
        traj_collect = self.gt_data['traj']
        times_collect = self.gt_data['times']
        data_collect = self.gt_data['data']
        cov_collect = self.gt_data['covs']
        save_dir = os.path.join(self.save_dir, "latent_trajectories")
        os.makedirs(save_dir, exist_ok=True)

        for subject_idx, sub in enumerate(zip(data_collect, cov_collect, times_collect, traj_collect)):
            data, cov, times, traj_true = sub
            t_np = self._to_numpy(times)
            K = traj_true.shape[1]

            save_dir = os.path.join(self.save_dir, "latent_trajectories")
            os.makedirs(save_dir, exist_ok=True)

            cols = 3
            rows = math.ceil(K / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(8, 2.5 * rows), sharex=True, constrained_layout=True)
            axes_flat = np.atleast_1d(axes).flatten()

            for k in range(K):
                ax = axes_flat[k]
                # Plot Ground Truth
                ax.plot(t_np, traj_true[:, k], color='black', linestyle='--', lw=1.5, label='Ground Truth', alpha=0.8)
                
                for m_idx, (model_name, runs) in enumerate(models.items()):
                    trajs = []
                    for run in runs:
                        trajs.append(self._latent_trajectory_estimation(run, data, times, cov))
                    
                    trajs = np.array(trajs) # [Runs, T, K]
                    mean = np.mean(trajs, axis=0)[:, k]
                    std = np.std(trajs, axis=0)[:, k]
                    
                    color = self.palette[m_idx % len(self.palette)]
                    ax.plot(t_np, mean, label=model_name, color=color)
                    # print(f"Subject {subject_idx}, Factor {k}: mean shape {mean.shape}, std shape {std.shape}")
                    ax.fill_between(t_np, mean - std, mean + std, color=color, alpha=0.15)

                ax.set_title(f"Latent Factor {k+1}", fontsize=10)
                sns.despine(ax=ax)
                if k == 0:
                    ax.legend(loc='upper right', frameon=False, fontsize=7)

            for j in range(k + 1, len(axes_flat)):
                axes_flat[j].axis('off')

            plt.savefig(os.path.join(save_dir, f"trajectories_subject_{subject_idx}.pdf"), bbox_inches='tight')
            plt.close()

    def plot_loading_recovery(self, models):
        """Heatmap comparison of Ground Truth and aligned estimated Loadings."""
        L_true_np = self._to_numpy(self.gt['Lambda'])
        save_dir = os.path.join(self.save_dir, "loadings")
        os.makedirs(save_dir, exist_ok=True)

        for model_name, runs in models.items():
            num_runs = len(runs)
            fig, axes = plt.subplots(num_runs, 3, figsize=(10, 3 * num_runs), squeeze=False, constrained_layout=True)
            
            # Setup shared color limits for better comparison
            v_min, v_max = L_true_np.min(), L_true_np.max()

            for idx, run in enumerate(runs):
                R = self._compute_alignment(run)
                L_aligned = (run.get_Lambda().detach() @ R).cpu().numpy()
                residual = L_true_np - L_aligned

                # Column 1: True
                sns.heatmap(L_true_np, ax=axes[idx, 0], cmap='viridis', vmin=v_min, vmax=v_max, cbar=idx==0)
                axes[idx, 0].set_title(r"True $\Lambda$" if idx==0 else "")
                
                # Column 2: Aligned Est
                sns.heatmap(L_aligned, ax=axes[idx, 1], cmap='viridis', vmin=v_min, vmax=v_max, cbar=idx==0)
                axes[idx, 1].set_title(f"{model_name} (Run {idx})" if idx==0 else f"Run {idx}")
                
                # Column 3: Residual
                res_max = np.abs(residual).max()
                sns.heatmap(residual, ax=axes[idx, 2], cmap='RdBu_r', center=0, vmin=-res_max, vmax=res_max, cbar=idx==0)
                axes[idx, 2].set_title("Residual" if idx==0 else "")

                for ax in axes[idx]:
                    ax.set_xticks([]); ax.set_yticks([])

            plt.savefig(os.path.join(save_dir, f"loading_grid_{model_name}.pdf"), bbox_inches='tight')
            plt.close()

    def plot_lambda_comparison(self, models):
            """
            Visualizes the true loading matrix vs the aligned estimated matrix.
            Optimized for publication with shared scales and professional color maps.
            """
            L_true_np = self._to_numpy(self.gt['Lambda'])
            
            # Shared color limits based on Ground Truth to ensure visual comparability
            v_min, v_max = L_true_np.min(), L_true_np.max()

            for model_name, runs in models.items():
                for idx, run in enumerate(runs):
                    # Calculate aligned estimate
                    R = self._compute_alignment(run)
                    L_est = run.get_Lambda().detach()
                    L_aligned = (L_est @ R).cpu().numpy()

                    # Use a slightly smaller figure size (standard 2-column paper width)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
                    
                    # Plot Ground Truth
                    im1 = ax1.imshow(L_true_np, aspect='auto', cmap='viridis', vmin=v_min, vmax=v_max)
                    ax1.set_title(r"Ground Truth $\Lambda$", fontweight='bold')
                    ax1.set_ylabel("Features (D)")
                    ax1.set_xlabel("Latent Factors (K)")

                    # Plot Aligned Estimate
                    im2 = ax2.imshow(L_aligned, aspect='auto', cmap='viridis', vmin=v_min, vmax=v_max)
                    ax2.set_title(r"Aligned Estimated $\Lambda$\n" + f"({model_name}, Run {idx})", fontweight='bold')
                    ax2.set_xlabel("Latent Factors (K)")

                    # Single colorbar to indicate shared scale
                    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.05)
                    cbar.ax.set_ylabel('Weight Intensity', rotation=-90, va="bottom")

                    # Clean up ticks for a cleaner aesthetic
                    for ax in [ax1, ax2]:
                        ax.tick_params(axis='both', which='both', length=0)
                        sns.despine(ax=ax, left=True, bottom=True)

                    # Save as PDF for vector-quality scaling in papers
                    save_path = os.path.join(self.save_dir, f"lambda_comp_{model_name}_run_{idx}.pdf")
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()