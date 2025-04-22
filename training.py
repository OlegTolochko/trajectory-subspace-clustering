import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from models.subspace_estimator import SubspaceEstimator
from losses import L_FeatDiff, L_InfoNCE, L_Residual
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self):
        self.x_data
        self.labels
        self.t_data

    def __len__(self):
        return len(self.x)

    def __getiem__(self, idx):
        x = None
        label = None
        return x, label

def prepare_data():
    return None

def main():
    return None
    