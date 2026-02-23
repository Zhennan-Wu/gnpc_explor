import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg
import numpy as np


def visualize_results(model, L_true, data, times, covs):
    # 1. EM Convergence Plots [cite: 406, 358-359]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(model.history['mse'], color='blue', label='MSE')
    axes[0].set_title("EM Convergence (Reconstruction Error)")
    axes[1].plot(model.history['corr'], color='green', label='Correlation')
    axes[1].set_title("Factor Recovery (Correlation)")
    plt.savefig("em_convergence.png", dpi=300)
    plt.show()

    # 2. Factor Loading Heat Map [cite: 405, 406]
    L_true_np = L_true.numpy()
    L_est_np = model.Lambda.detach().numpy()
    U, _, Vt = scipy.linalg.svd(L_true_np.T @ L_est_np)
    L_aligned = L_est_np @ (Vt.T @ U.T)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(L_true_np, ax=axes[0], cmap='viridis', cbar=True)
    axes[0].set_title("|Lambda| True")
    sns.heatmap(np.abs(L_aligned), ax=axes[1], cmap='viridis', cbar=True)
    axes[1].set_title("|Lambda| Estimated (Aligned)")
    plt.savefig("factor_loadings.png", dpi=300)
    plt.show()

    # 3. Factor Temporal Comparison (Figure 4) [cite: 605]
    subject_idx = 0
    f_smooth, _, _ = model.kalman_filter_smoother(data[subject_idx], times[subject_idx], covs[subject_idx])
    t = times[subject_idx].numpy()
    f_smooth = f_smooth.detach().numpy()
    
    plt.figure(figsize=(10, 4))
    for k in range(model.K):
        plt.plot(t, f_smooth[:, k], label=f'Factor {k}', marker='o')
    plt.title(f"Latent Trajectories for Subject {subject_idx}")
    plt.xlabel("Time t")
    plt.ylabel("Latent State f(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("latent_trajectories.png", dpi=300)
    plt.show()