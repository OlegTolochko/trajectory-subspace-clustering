import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inference import load_model


def test_orthogonality(save_plots=True, output_dir="out/plots/", tick_frequency=4, font_size=18):
    model = load_model()
    subspace_estimator = model.subspace_estimator
    device = next(subspace_estimator.parameters()).device

    seq_length = 300
    t_vector = torch.arange(seq_length, dtype=torch.float32).to(device)
    t_vector_batch = t_vector.unsqueeze(0)

    with torch.no_grad():
        h_t_values = subspace_estimator.calculate_basis_functions(t_vector_batch)

    basis_function_vectors = h_t_values.squeeze(0).cpu().numpy()
    N_basis_functions = basis_function_vectors.shape[1]
    print(f"Shape of basis function matrix (F, N): {basis_function_vectors.shape}")

    normalized_basis_vectors = basis_function_vectors / (
        np.linalg.norm(basis_function_vectors, axis=0, keepdims=True) + 1e-9
    )
    cosine_sim_matrix = normalized_basis_vectors.T @ normalized_basis_vectors
    sim_matrix = basis_function_vectors.T @ basis_function_vectors

    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)

    tick_positions = range(0, N_basis_functions, tick_frequency)
    tick_labels = [str(i) for i in tick_positions]

    # Gramian Matrix:
    plt.figure(figsize=(12, 10))
    ax1 = sns.heatmap(sim_matrix, cmap="coolwarm", center=0)
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=font_size)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=font_size)
    
    plt.title("Gramian Matrix (Dot Products) of Basis Functions", fontsize=font_size+2)
    plt.xlabel("Basis Function Index", fontsize=font_size)
    plt.ylabel("Basis Function Index", fontsize=font_size)
    
    cbar = ax1.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)

    if save_plots:
        plt.savefig(
            f"{output_dir}gramian_matrix_seq_len_{seq_length}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {output_dir}gramian_matrix_seq_len_{seq_length}.png")

    plt.show()

    # Cosine Similarity Matrix:
    plt.figure(figsize=(12, 10))
    ax2 = sns.heatmap(cosine_sim_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=font_size)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=font_size)
    
    plt.title(f"Model 1: Cosine Similarity for Seq. len of {seq_length}", fontsize=font_size+2)
    plt.xlabel("Basis function Index", fontsize=font_size)
    plt.ylabel("Basis function Index", fontsize=font_size)
    
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)

    if save_plots:
        plt.savefig(
            f"{output_dir}cosine_similarity_seq_len_{seq_length}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {output_dir}cosine_similarity_seq_len_{seq_length}.png")

    plt.show()

    diag_mask = ~np.eye(N_basis_functions, dtype=bool)
    mean_abs_off_diagonal_cosine_sim = np.mean(np.abs(cosine_sim_matrix[diag_mask]))
    print(
        f"Mean absolute off-diagonal cosine similarity: {mean_abs_off_diagonal_cosine_sim:.4f}"
    )


if __name__ == "__main__":
    test_orthogonality(save_plots=True, output_dir="out/plots/")
