import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from models.subspace_estimator import SubspaceEstimator
from losses import L_FeatDiff, L_InfoNCE, L_Residual
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import os
import numpy as np
from sklearn.model_selection import train_test_split
from inference import compare_all_clustering_methods, evaluate_model_performance
from datasets import Hopkins155, KT3DMoSeg
from tqdm import tqdm
import optuna

def reconstruct_x(x_original, B_estimated):
    try:
        batch_size, seq_len, _ = x_original.shape
        x_flattend = x_original.reshape(batch_size, 2*seq_len, 1)
        
        B_dagger = torch.linalg.pinv(B_estimated)
        c = torch.bmm(B_dagger, x_flattend)
        x_reconst_flat = torch.bmm(B_estimated, c)
        x_reconst = x_reconst_flat.reshape(batch_size, seq_len, 2)
    except:
        print("Error occured in x reconstruction.")
    return x_reconst

def train_model(train_set, val_set,
                        device,
                        lr, weight_decay_val, scheduler_gamma,
                        w_info, w_res, w_feat,
                        pretraining_epochs, full_epochs):
    full_model = TrajectoryEmbeddingModel().to(device)
    optimizer_stage1 = optim.Adam(full_model.feature_extractor.parameters(), lr=lr, weight_decay=weight_decay_val)
    optimizer_stage2 = optim.Adam(full_model.parameters(), lr=lr, weight_decay=weight_decay_val)
    scheduler_stage1 = ExponentialLR(optimizer_stage1, gamma=scheduler_gamma)
    scheduler_stage2 = ExponentialLR(optimizer_stage2, gamma=scheduler_gamma)
    
    train_loader = DataLoader(train_set,
                        batch_size=1,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        persistent_workers=True if os.name == 'nt' else False)
    
    print(f"Starting HPO trial: lr={lr:.2e}, wd={weight_decay_val:.2e}, gamma={scheduler_gamma:.4f}, w_info={w_info:.2f}, w_res={w_res:.2f}, w_feat={w_feat:.2f}")
    
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
            mask = torch.rand_like(seq_x[..., :1]) > 0.25  # 25 % dropout
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
            
            loss_infoNCE = L_InfoNCE(v_norm, seq_labels)
            loss_residual = L_Residual(x_original=seq_x, x_reconstructed=x_recostructed)
            
            f_reconstructed = full_model.feature_extractor(x_recostructed_permuted)
            loss_featdiff = L_FeatDiff(f_original=f, f_reconstructed=f_reconstructed)
            

            total_loss = (w_info * loss_infoNCE + w_res * loss_residual + w_feat * loss_featdiff)
            total_loss.backward()
            optimizer_stage2.step()
            epoch_loss_stage2 += total_loss.item()
            num_seq_processed += 1
        
        scheduler_stage2.step()
        avg_epoch_loss = epoch_loss_stage2 / num_seq_processed if num_seq_processed > 0 else 0.0
        print(f"Full Training Epoch {epoch + 1}/{full_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

    full_model.eval()
    metric = evaluate_model_performance(model=full_model, data=val_set, 
                                                    cluster_algo_name='hierarchical',
                                                    device_str=device.type)

    return metric


def create_and_train_final_model(train_set,
                                 device,
                                 lr, weight_decay_val, scheduler_gamma,
                                 w_info, w_res, w_feat,
                                 pretraining_epochs, full_epochs):
    print("\n--- Training Final Model with Best Hyperparameters ---")
    print(f"Parameters: lr={lr:.2e}, wd={weight_decay_val:.2e}, gamma={scheduler_gamma:.4f}")
    print(f"Loss Weights: w_info={w_info:.2f}, w_res={w_res:.2f}, w_feat={w_feat:.2f}")
    print(f"Epochs: Pretrain={pretraining_epochs}, Full={full_epochs}")

    final_model = TrajectoryEmbeddingModel().to(device)
    optimizer_stage1 = optim.Adam(final_model.feature_extractor.parameters(), lr=lr, weight_decay=weight_decay_val)
    optimizer_stage2 = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay_val)
    scheduler_stage1 = ExponentialLR(optimizer_stage1, gamma=scheduler_gamma)
    scheduler_stage2 = ExponentialLR(optimizer_stage2, gamma=scheduler_gamma)

    train_loader = DataLoader(train_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True if os.name == 'nt' else False)

    for epoch in tqdm(range(pretraining_epochs), desc="Final Model Pretraining"):
        epoch_loss_stage1 = 0.0
        num_seq_processed = 0
        final_model.feature_extractor.train()
        final_model.subspace_estimator.eval()
        for batch_data in train_loader:
            seq_x = batch_data['trajectories'].to(device).squeeze(0)
            seq_labels = batch_data['labels'].to(device).squeeze(0)
            num_points = seq_x.shape[0]
            if num_points <= 1: continue
            
            optimizer_stage1.zero_grad()
            mask = torch.rand_like(seq_x[..., :1]) > 0.25
            seq_x_aug = seq_x * mask
            x_permuted = seq_x_aug.permute(0, 2, 1) 
            f = final_model.feature_extractor(x_permuted)
            loss = L_InfoNCE(f, seq_labels)
            loss.backward()
            optimizer_stage1.step()
            epoch_loss_stage1 += loss.item()
            num_seq_processed += 1
        scheduler_stage1.step()
        avg_epoch_loss = epoch_loss_stage1 / num_seq_processed if num_seq_processed > 0 else 0.0
        print(f"Final Pretraining Epoch {epoch + 1}/{pretraining_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

    for epoch in tqdm(range(full_epochs), desc="Final Model Full Training"):
        final_model.train()
        epoch_loss_stage2 = 0.0
        num_seq_processed = 0
        for batch_data in train_loader:
            seq_x = batch_data['trajectories'].to(device).squeeze(0)
            seq_labels = batch_data['labels'].to(device).squeeze(0)
            seq_t = batch_data['times'].to(device).squeeze(0)
            num_points = seq_x.shape[0]
            
            optimizer_stage2.zero_grad()
            f, B = final_model(seq_x, seq_t)
            
            B_flat = B.view(num_points, -1)
            v = torch.cat((f, B_flat), dim=1)
            v_norm = F.normalize(v, p=2, dim=1)
            
            x_reconstructed = reconstruct_x(seq_x, B)
            x_reconstructed_permuted = x_reconstructed.permute(0, 2, 1)
            
            loss_infoNCE = L_InfoNCE(v_norm, seq_labels)
            loss_residual = L_Residual(x_original=seq_x, x_reconstructed=x_reconstructed)
            
            f_reconstructed = final_model.feature_extractor(x_reconstructed_permuted)
            loss_featdiff = L_FeatDiff(f_original=f, f_reconstructed=f_reconstructed)
            
            total_loss = (w_info * loss_infoNCE + w_res * loss_residual + w_feat * loss_featdiff)
            total_loss.backward()
            optimizer_stage2.step()
            epoch_loss_stage2 += total_loss.item()
            num_seq_processed += 1
        scheduler_stage2.step()
        avg_epoch_loss = epoch_loss_stage2 / num_seq_processed if num_seq_processed > 0 else 0.0
        print(f"Final Full Training Epoch {epoch + 1}/{full_epochs}, Avg Loss: {avg_epoch_loss:.4f}")

    final_model.eval()
    return final_model

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.001
    weight_decay_val = 1e-5
    scheduler_gamma = 0.999

    w_info = trial.suggest_float("w_info", 0.1, 2.0)
    w_res = trial.suggest_float("w_res", 0.5, 5.0)
    w_feat = trial.suggest_float("w_feat", 0.5, 5.0)

    pretraining_epochs = 10
    full_epochs = 25

    if not hasattr(objective, 'train_dataset'):
        print("Loading dataset for HPO study...")
        objective.train_dataset = Hopkins155()
        seq_ids = list(range(len(objective.train_dataset)))
        train_ids, val_ids = train_test_split(seq_ids, test_size=0.2, random_state=42, shuffle=True)
        objective.train_set = torch.utils.data.Subset(objective.train_dataset, train_ids)
        objective.val_set   = torch.utils.data.Subset(objective.train_dataset, val_ids)
        print("Dataset loaded and split.")

    train_set = objective.train_set
    val_set = objective.val_set

    validation_metric = train_model(
        train_set, val_set, device,
        lr, weight_decay_val, scheduler_gamma,
        w_info, w_res, w_feat,
        pretraining_epochs, full_epochs
    )

    trial.report(validation_metric, step=full_epochs)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return validation_metric

def original_main():
    lr = 0.001
    weight_decay = 1e-5
    scheduler_gamma = 0.999
    w_info = 0.5
    w_res = 1.0
    w_feat = 1.0
    pretraining_epochs = 50
    full_epochs = 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running original_main with device: {device}")

    train_dataset_orig = KT3DMoSeg()
    seq_ids_orig = list(range(len(train_dataset_orig)))
    train_ids_orig, val_ids_orig = train_test_split(seq_ids_orig, test_size=0.2, random_state=42, shuffle=True)
    train_set_orig = torch.utils.data.Subset(train_dataset_orig, train_ids_orig)
    val_set_orig   = torch.utils.data.Subset(train_dataset_orig, val_ids_orig)

    print("Starting model training with fixed hyperparameters...")
    model = create_and_train_final_model(
        train_set_orig, device,
        lr, weight_decay, scheduler_gamma,
        w_info, w_res, w_feat,
        pretraining_epochs, full_epochs
    )
    
    model.eval()
    metric = evaluate_model_performance(model=model, data=val_set_orig, 
                                                cluster_algo_name='hierarchical',
                                                device_str=device)
    
    print("Model training complete.")
    model_save_path = 'out/models/kt_trained_model_weights_normalized7.pt'
    torch.save(model.state_dict(), model_save_path)

    print("Original main finished (evaluation/saving would happen here).")


def eval_model(model, val_set):
    compare_all_clustering_methods(model=model, data=val_set)
    
def main():
    run_hpo = False

    if run_hpo:
        print("Starting Hyperparameter Optimization with Optuna...")
        if hasattr(objective, 'train_dataset'):
            del objective.train_dataset

        db_url = "sqlite:///trajectory_hpo.db"
        study_name = "trajectory_embedding_study_v1"

        study = optuna.create_study(
            study_name=study_name,
            storage=db_url,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True
        )

        study.optimize(objective, n_trials=25)
        
        print("Best trial:")
        best_trial_obj = study.best_trial
        print(f"Best error rate: {best_trial_obj.value}")
        print("Params: ")
        for key, value in best_trial_obj.params.items():
            print(f"    {key}: {value}")

        df = study.trials_dataframe()
        df.to_csv("hpo_study_results.csv")

        print("\nTraining final model with best hyperparameters...")
        best_hps = best_trial_obj.params
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not hasattr(objective, 'train_dataset'):
            print("Reloading dataset for final model training...")
            objective.train_dataset = Hopkins155()
            seq_ids = list(range(len(objective.train_dataset)))
            train_ids, val_ids = train_test_split(seq_ids, test_size=0.2, random_state=42, shuffle=True)
            final_train_set = torch.utils.data.Subset(objective.train_dataset, train_ids)
            final_val_set = torch.utils.data.Subset(objective.train_dataset, val_ids)
        else:
            final_train_set = objective.train_set
            final_val_set = objective.val_set

        print(f"Re-running with best HPs: {best_hps}")

        final_pretraining_epochs = 50
        final_full_epochs = 200

        best_model = create_and_train_final_model(
            final_train_set,
            device,
            best_hps["lr"], best_hps["weight_decay"], best_hps["scheduler_gamma"],
            best_hps["w_info"], best_hps["w_res"], best_hps["w_feat"],
            final_pretraining_epochs,
            final_full_epochs
        )
        
        best_model.eval()
        metric = evaluate_model_performance(model=best_model, data=final_val_set, 
                                                    cluster_algo_name='hierarchical',
                                                    device_str=device)
        
        best_model_save_path = 'out/models/best_hpo_model.pt'
        torch.save(best_model.state_dict(), best_model_save_path)
        print(f"\nBest model saved to {best_model_save_path}")

    else:
        print("Running a single original training run...")
        original_main()
    
if __name__ == '__main__':
    main()
