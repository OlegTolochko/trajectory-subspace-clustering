import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from models.subspace_estimator import SubspaceEstimator
from losses import L_FeatDiff, L_InfoNCE, L_Residual
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import scipy
import numpy as np


def reconstruct_x(x_original, B_estimated):
    batch_size, seq_len, _ = x_original.shape
    x_flattend = x_original.reshape(batch_size, 2*seq_len, 1)
    
    B_dagger = torch.linalg.pinv(B_estimated)
    c = torch.bmm(B_dagger, x_flattend)
    x_reconst_flat = torch.bmm(B_estimated, c)
    x_reconst = x_reconst_flat.reshape(batch_size, seq_len, 2)
    return x_reconst

def train_model(batch_size=64, pretraining_epochs=50, full_epochs=50, learning_rate=0.001):
    """
    Assumed training_samples structure: (x, t, labels)
    """
    full_model = TrajectoryEmbeddingModel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = full_model.to(device)
    optimizer_stage1 = optim.Adam(full_model.feature_extractor.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(full_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    train_dataset = Hopkins155()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # pretraining:
    for epoch in range(pretraining_epochs):
        epoch_loss = 0
        full_model.feature_extractor.train()
        full_model.subspace_estimator.eval()
        for batch_data in train_loader:
            batch_x, batch_t, batch_lables = batch_data
            
            optimizer_stage1.zero_grad()
            f_vectors = full_model.feature_extractor(batch_x)
            loss = L_InfoNCE(f_vectors, batch_lables)
            loss.backward()
            optimizer_stage1.step()
            epoch_loss += loss.item()
            
        print(f"Pretraining Epoch {epoch + 1}/{pretraining_epochs}, "
              f"Loss: {epoch_loss / len(train_dataset):.4f}")
        
    # full model training:    
    for i in range(full_epochs):
        full_model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            batch_x, batch_t, batch_lables = batch_data
            
            optimizer.zero_grad()
            f, B = full_model(batch_x, batch_t)
            
            x_recostructed = reconstruct_x(batch_x, B)
            loss_infoNCE = L_InfoNCE(f, batch_lables)
            loss_residual = L_Residual(x_original=batch_x, x_recostructed=x_recostructed)
            
            f_reconstructed = full_model.feature_extractor(x_recostructed)
            loss_featdiff = L_FeatDiff(f_original=f, f_reconstructed=f_reconstructed)
            
            w_info = 1.0
            w_res = 0.5
            w_feat = 0.5

            total_loss = (w_info * loss_infoNCE + w_res * loss_residual + w_feat * loss_featdiff)
            total_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Pretraining Epoch {epoch + 1}/{full_epochs}, "
              f"Loss: {total_loss / len(train_dataset):.4f}")
    
    return None

class Hopkins155(Dataset):
    def __init__(self, root_dir="data/Hopkins155/"):
        self.root_dir = root_dir
        self.sequence_data = []
        
        print(f"Loading Hopkins155 data from: {root_dir}")
        for seq_name in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_name)
            if os.path.isdir(seq_path):
                mat_file_name = f"{seq_name}_truth.mat"
                mat_file_path = os.path.join(seq_path, mat_file_name)
            
            try:
                mat_data = scipy.io.loadmat(mat_file_path)
                x_data_load = None
                if 'x' in mat_data:
                    x_data_load = mat_data['x']
                    print(f"Using 'x' data (shape {x_data_load.shape}) for {seq_name}")
                
                coords_2PF = x_data_load[0:2, :, :] # Shape (2, P, F)
                num_points = coords_2PF.shape[1]
                num_frames = coords_2PF.shape[2]
                trajectories = np.transpose(coords_2PF, (1, 2, 0))
                time_vectors = torch.arange(num_frames).unsqueeze(0).repeat(num_points, 1)
                
                if 's' in mat_data:
                    labels_load = mat_data['s'].reshape(-1)

                self.sequence_data.append({
                    'name': seq_name,
                    'trajectories': trajectories.astype(np.float32),
                    'times': time_vectors,
                    'labels': labels_load.astype(np.int64)
                })
                
            except Exception as e:
                print(f"Error loading or processing {mat_file_path}: {e}")
                
        print(f"finished loading data for {len(self.sequence_data)} sequences")

    def __len__(self):
        return len(self.sequence_data)

    def __getiem__(self, idx):
        if idx >= len(self.sequence_data):
            raise IndexError("Index out of bounds")

        seq_info = self.sequence_data[idx]
        trajectories = seq_info['trajectories']
        labels = seq_info['labels']
        seq_name = seq_info['name']
        time_vectors = seq_info['times']
        
        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        time_tensor = time_vectors.long()
        
        return {
            'trajectories': trajectories_tensor,
            'labels': labels_tensor,
            'times': time_tensor,
            'name': seq_name
        }

def prepare_data():
    return None

def main():
    return None
    
if __name__ == '__main__':
    None
