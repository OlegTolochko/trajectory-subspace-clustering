import torch
import numpy as np
import os
from models.trajectory_embedder import TrajectoryEmbeddingModel


def extract_subspace_parameters(model_path, output_path="subspace_parameters.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = TrajectoryEmbeddingModel() 

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}")

    subspace_estimator = model.subspace_estimator

    parameters_to_save = {}
    param_names = ['mu_basis', 'alpha_basis', 'beta_basis', 'gamma_basis']

    for param_name in param_names:
        if hasattr(subspace_estimator, param_name):
            param_tensor = getattr(subspace_estimator, param_name)
            if isinstance(param_tensor, torch.nn.Parameter):
                param_numpy = param_tensor.data.cpu().numpy()
                parameters_to_save[param_name] = param_numpy
                print(f"Extracted '{param_name}' with shape: {param_numpy.shape}")

    np.savez(output_path, **parameters_to_save)


if __name__ == '__main__':
    trained_model = 'out/models/trained_model_weights_normalized.pt' 
    output_file = 'extracted_subspace_basis_parameters.npz'

    extract_subspace_parameters(trained_model, output_file)