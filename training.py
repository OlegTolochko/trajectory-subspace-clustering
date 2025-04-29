import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from models.subspace_estimator import SubspaceEstimator
from losses import L_FeatDiff, L_InfoNCE, L_Residual
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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

def train_model(batch_size=1, pretraining_epochs=10, full_epochs=20, learning_rate=0.001):
    full_model = TrajectoryEmbeddingModel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    full_model = full_model.to(device)
    optimizer_stage1 = optim.Adam(full_model.feature_extractor.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_stage2 = optim.Adam(full_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    train_dataset = Hopkins155()
    train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        persistent_workers=True if os.name == 'nt' else False)

    # pretraining:
    for epoch in range(pretraining_epochs):
        epoch_loss_stage1 = 0.0
        num_seq_processed = 0
        full_model.feature_extractor.train()
        full_model.subspace_estimator.eval()
        for batch_data in train_loader:
            seq_x = batch_data['trajectories'].to(device).squeeze(0)
            seq_labels = batch_data['labels'].to(device).squeeze(0)
            num_points = seq_x.shape[0]
            if num_points <= 1: continue
            
            optimizer_stage1.zero_grad()
            # model input: (Batch=P, Channels=2, SeqLen=F)
            x_permuted = seq_x.permute(0, 2, 1) # (P, 2, F)
            f = full_model.feature_extractor(x_permuted)
            # normalization of the f vectors could be performed here
            loss = L_InfoNCE(f, seq_labels)
            loss.backward()
            optimizer_stage1.step()
            epoch_loss_stage1 += loss.item()
            num_seq_processed += 1
            
        avg_epoch_loss = epoch_loss_stage1 / num_seq_processed if num_seq_processed > 0 else 0.0
        print(f"Pretraining Epoch {epoch + 1}/{pretraining_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

        
    # full model training:    
    for epoch in range(full_epochs):
        full_model.train()
        epoch_loss_stage2 = 0.0
        num_seq_processed = 0
        for batch_data in train_loader:
            seq_x = batch_data['trajectories'].to(device).squeeze(0) # (P, F, 2)
            seq_labels = batch_data['labels'].to(device).squeeze(0) # (P,)
            seq_t = batch_data['times'].to(device).squeeze(0)  # (P, F)
            num_points = seq_x.shape[0]
            
            optimizer_stage2.zero_grad()
            f, B = full_model(seq_x, seq_t)
            
            B_flat = B.view(num_points, -1) # (P, 2F*rank)
            v = torch.cat((f, B_flat), dim=1)
            v_norm = F.normalize(v, p=2, dim=1)
            
            x_recostructed = reconstruct_x(seq_x, B) # (P, F, 2)
            x_recostructed_permuted = x_recostructed.permute(0, 2, 1)
            
            loss_infoNCE = L_InfoNCE(v, seq_labels)
            loss_residual = L_Residual(x_original=seq_x, x_reconstructed=x_recostructed)
            
            f_reconstructed = full_model.feature_extractor(x_recostructed_permuted)
            loss_featdiff = L_FeatDiff(f_original=f, f_reconstructed=f_reconstructed)
            
            w_info = 1.0
            w_res = 0.5
            w_feat = 0.5

            total_loss = (w_info * loss_infoNCE + w_res * loss_residual + w_feat * loss_featdiff)
            total_loss.backward()
            optimizer_stage2.step()
            epoch_loss_stage2 += total_loss.item()
            num_seq_processed += 1
        
        avg_epoch_loss = epoch_loss_stage2 / num_seq_processed if num_seq_processed > 0 else 0.0
        print(f"Full Training Epoch {epoch + 1}/{full_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

    return full_model

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
                
                coords_2PF = x_data_load[0:2, :, :] # (2, P, F)
                num_points = coords_2PF.shape[1]
                num_frames = coords_2PF.shape[2]
                trajectories = np.transpose(coords_2PF, (1, 2, 0)) # (P, F, 2)
                base_time = torch.arange(num_frames)
                time_vectors = base_time.expand(num_points, -1)
                
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

    def __getitem__(self, idx):
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
        num_clusters = len(torch.unique(labels_tensor))
        
        return {
            'trajectories': trajectories_tensor,
            'labels': labels_tensor,
            'times': time_tensor,
            'name': seq_name,
            'num_clusters': num_clusters
        }

def main():
    trained_model = train_model()
    if trained_model:
        print("Model training complete.")
    else:
        print("Model training failed.")
    
    pytorch_save_path = 'out/models/trained_model_weights.pt'
    print(f"Saving model state_dict to {pytorch_save_path}...")
    torch.save(trained_model.state_dict(), pytorch_save_path)
    print("Saved.")
    
if __name__ == '__main__':
    main()
