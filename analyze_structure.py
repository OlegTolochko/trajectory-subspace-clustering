import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inference import load_model

def test_orthogonality():
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

    normalized_basis_vectors = basis_function_vectors / (np.linalg.norm(basis_function_vectors, axis=0, keepdims=True) + 1e-9)
    cosine_sim_matrix = normalized_basis_vectors.T @ normalized_basis_vectors
    sim_matrix = basis_function_vectors.T @ basis_function_vectors

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='coolwarm', center=0)
    plt.title("Gramian Matrix (Dot Products) of Basis Functions")
    plt.xlabel("Basis Function Index")
    plt.ylabel("Basis Function Index")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f"Model 2: Cosine Similarity for Seq. len of {seq_length}")
    plt.xlabel("Basis function Index")
    plt.ylabel("Basis function Index")
    plt.show()
    
    diag_mask = ~np.eye(N_basis_functions, dtype=bool) 
    mean_abs_off_diagonal_cosine_sim = np.mean(np.abs(cosine_sim_matrix[diag_mask]))
    print(f"Mean absolute off-diagonal cosine similarity: {mean_abs_off_diagonal_cosine_sim:.4f}")

if __name__ == '__main__':
    test_orthogonality()
    
    
        