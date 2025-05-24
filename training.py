import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from models.subspace_estimator import SubspaceEstimator
from losses import L_FeatDiff, L_InfoNCE, L_Residual, L_orthogonal
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import os
import numpy as np
from sklearn.model_selection import train_test_split
from inference import compare_all_clustering_methods
from datasets import Hopkins155, KT3DMoSeg
from tqdm import tqdm

def reconstruct_x(x_original, B_estimated):
    with torch.no_grad():
        try:
            batch_size, seq_len, _ = x_original.shape
            x_flattend = x_original.reshape(batch_size, 2*seq_len, 1)
            solution = torch.linalg.lstsq(B_estimated, x_flattend)
            c = solution.solution
            
            x_reconst_flat = torch.bmm(B_estimated, c)
            x_reconst = x_reconst_flat.reshape(batch_size, seq_len, 2)
        except Exception as e:
            print(f"Error occurred in x reconstruction: {e}")
    return x_reconst

def train_model(train_set, batch_size=1, pretraining_epochs=100, full_epochs=200, learning_rate=0.001):
    full_model = TrajectoryEmbeddingModel()
    include_ortho_loss = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    full_model = full_model.to(device)
    optimizer_stage1 = optim.Adam(full_model.feature_extractor.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_stage2 = optim.Adam(full_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler_stage1 = ExponentialLR(optimizer_stage1, gamma=0.999)
    scheduler_stage2 = ExponentialLR(optimizer_stage2, gamma=0.999)
    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    # pretraining:
    for epoch in tqdm(range(pretraining_epochs), desc="Pretraining Model"):
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
            mask = torch.rand_like(seq_x[..., :1], device=device) > 0.25  # 25 % dropout
            seq_x = seq_x * mask
            # model input: (Batch=P, Channels=2, SeqLen=F)
            x_permuted = seq_x.permute(0, 2, 1) # (P, 2, F)
            f = full_model.feature_extractor(x_permuted)
            loss = L_InfoNCE(f, seq_labels)
            loss.backward()
            optimizer_stage1.step()
            epoch_loss_stage1 += loss.item()
            num_seq_processed += 1
        
        scheduler_stage1.step()
        avg_epoch_loss = epoch_loss_stage1 / num_seq_processed if num_seq_processed > 0 else 0.0
        print(f"Pretraining Epoch {epoch + 1}/{pretraining_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

        
    # full model training:    
    for epoch in tqdm(range(full_epochs), desc="Training Full Model"):
        full_model.train()
        epoch_loss_stage2 = 0.0
        num_seq_processed = 0
        w_info_sum = 0.0
        w_res_sum = 0.0
        w_feat_sum = 0.0
        w_ortho_sum = 0.0
        for batch_data in train_loader:
            seq_x = batch_data['trajectories'].to(device).squeeze(0) # (P, F, 2)
            seq_labels = batch_data['labels'].to(device).squeeze(0) # (P,)
            seq_t = batch_data['times'].to(device).squeeze(0)  # (P, F)
            num_points = seq_x.shape[0]
            
            optimizer_stage2.zero_grad()
            f, B, h_t = full_model(seq_x, seq_t)
            
            B_flat = B.view(num_points, -1) # (P, 2F*rank)
            v = torch.cat((f, B_flat), dim=1)
            v_norm = F.normalize(v, p=2, dim=1)
            
            x_recostructed = reconstruct_x(seq_x, B) # (P, F, 2)
            x_recostructed_permuted = x_recostructed.permute(0, 2, 1)
            
            loss_infoNCE = L_InfoNCE(v_norm, seq_labels)
            loss_residual = L_Residual(x_original=seq_x, x_reconstructed=x_recostructed)
            
            f_reconstructed = full_model.feature_extractor(x_recostructed_permuted)
            loss_featdiff = L_FeatDiff(f_original=f, f_reconstructed=f_reconstructed)
            if include_ortho_loss:
                loss_ortho = L_orthogonal(h_t)
            else:
                loss_ortho = torch.tensor(0.0, device=device)
            
            w_info = 1.0
            w_res = 1.0
            w_feat = 0.0
            w_ortho = 0.01

            w_info_sum += w_info * loss_infoNCE
            w_res_sum += w_res * loss_residual
            w_feat_sum += w_feat * loss_featdiff
            w_ortho_sum += w_ortho*loss_ortho
            
            total_loss = (w_info * loss_infoNCE + w_res * loss_residual + w_feat * loss_featdiff + w_ortho*loss_ortho)
            total_loss.backward()
            optimizer_stage2.step()
            epoch_loss_stage2 += total_loss.item()
            num_seq_processed += 1
        
        scheduler_stage2.step()
        avg_epoch_loss = epoch_loss_stage2 / num_seq_processed
        mean_w_info = w_info_sum / num_seq_processed
        mean_w_res = w_res_sum / num_seq_processed
        mean_w_feat = w_feat_sum / num_seq_processed
        mean_w_ortho = w_ortho_sum / num_seq_processed
        print(f"Full Training Epoch {epoch + 1}/{full_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch + 1}/{full_epochs}: InfoNCE Loss: {mean_w_info:.4f}, Res Loss: {mean_w_res:.4f}, Feat Loss: {mean_w_feat:.4f}, Ortho Loss: {mean_w_ortho:.4f}")

    return full_model

def eval_model(model, val_set):
    compare_all_clustering_methods(model=model, data=val_set)
    
def main():
    train_dataset = Hopkins155()

    seq_ids = list(range(len(train_dataset)))
    train_ids, val_ids = train_test_split(seq_ids, test_size=0.2, random_state=42, shuffle=True)
    train_set = torch.utils.data.Subset(train_dataset, train_ids)
    val_set   = torch.utils.data.Subset(train_dataset, val_ids)
    
    trained_model = train_model(train_set=train_set)
    if trained_model:
        print("Model training complete.")
    else:
        print("Model training failed.")
    
    eval_model(model=trained_model, val_set=val_set)

    pytorch_save_path = 'out/models/hopk155_100_200_ortho.pt'
    print(f"Saving model state_dict to {pytorch_save_path}...")
    torch.save(trained_model.state_dict(), pytorch_save_path)
    print("Saved.")
    
if __name__ == '__main__':
    main()
