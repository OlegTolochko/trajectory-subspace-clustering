import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from models.subspace_estimator import SubspaceEstimator

def reconstruct_x(x_original, B_estimated):
    batch_size, seq_len, _ = x_original.shape
    x_flattend = x_original.reshape(batch_size, 2*seq_len, 1)
    
    B_dagger = torch.linalg.pinv(B_estimated)
    c = torch.bmm(B_dagger, x_flattend)
    x_reconst_flat = torch.bmm(B_estimated, c)
    x_reconst = x_reconst_flat.reshape(batch_size, seq_len, 2)
    return x_reconst

def train_model(training_samples, batch_size=64, epochs=50, learning_rate=0.001):
    subspace_estimator = SubspaceEstimator()
    full_model = TrajectoryEmbeddingModel()
    
    return None

def prepare_data():
    return None

def main():
    return None
    