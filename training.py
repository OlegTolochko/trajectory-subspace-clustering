from datetime import datetime
import os
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from models.trajectory_embedder import TrajectoryEmbeddingModel
from losses import L_FeatDiff, L_InfoNCE, L_Residual, L_orthogonal
from datasets import (
    Hopkins155,
    KT3DMoSeg,
    Hopkins12,
    shift_seq,
    augment_normalized_data,
)
from inference import evaluate_model_performance


def train_model(config, model_save_path="./out/models/"):
    metrics_to_log = {
        "total_loss": 0,
        "infonce_loss": 0,
        "residual_loss": 0,
        "feat_diff_loss": 0,
        "ortho_loss": 0,
        f"mean_clustering_error": 0,
    }

    main_dataset = load_dataset(config["train_data"])
    main_train_loader, main_val_loader = get_train_val_loaders(
        main_dataset, config
    )
    additional_val_loader = None
    if config["additional_val_data"]:
        metrics_to_log.update(
            {f"{str.lower(config['additional_val_data'])}_mean_clustering_error": 0}
        )
        additional_val_dataset = load_dataset(config["additional_val_data"])
        additional_val_loader = DataLoader(
            additional_val_dataset, batch_size=config["batch_size"], shuffle=True
        )

    device = torch.device(config["device"])

    model = TrajectoryEmbeddingModel(alph0=config["alph0"])
    model = model.to(device)

    optimizer_stage1 = optim.Adam(
        model.feature_extractor.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    optimizer_stage2 = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler_stage1 = ExponentialLR(optimizer_stage1, gamma=config["scheduler_gamma"])
    scheduler_stage2 = ExponentialLR(optimizer_stage2, gamma=config["scheduler_gamma"])

    torch.backends.cudnn.benchmark = True

    pretrained_model = pretraining_loop(
        config=config,
        model=model,
        train_loader=main_train_loader,
        device=device,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
    )

    pretrained_model = pretrained_model.to(device)

    fully_trained_model = full_training_loop(
        config=config,
        model=pretrained_model,
        train_loader=main_train_loader,
        val_loader=main_val_loader,
        additional_val_loader=additional_val_loader,
        device=device,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        metrics=metrics_to_log,
    )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    wandb_id = wandb.run.id
    model_save_path = os.path.join(
        model_save_path, f"{config['model_name']}-{timestamp}-{wandb_id}.pth"
    )
    torch.save(fully_trained_model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")

    return fully_trained_model


def pretraining_loop(config, model, train_loader, device, optimizer, scheduler):
    for epoch in tqdm(range(config["pretraining_epochs"]), desc="Pretraining Model"):
        model = model.to(device)
        epoch_loss_stage1 = 0.0
        num_seq_processed = 0
        model.feature_extractor.train()
        model.subspace_estimator.eval()
        for batch_data in train_loader:
            seq_x = batch_data["trajectories"].to(device).squeeze(0)
            seq_labels = batch_data["labels"].to(device).squeeze(0)
            if (
                config["augmentation_max_shift_amount"] != 0.0
                or config["augmentation_shift_percent"] != 0.0
                or config["augmentation_occlusion_percent"] != 0
            ):
                seq_x = augment_normalized_data(
                    seq_x=seq_x,
                    max_shift_amount=config["augmentation_max_shift_amount"],
                    shift_percent=config["augmentation_shift_percent"],
                    occlusion_percent=config["augmentation_occlusion_percent"],
                )

            num_points = seq_x.shape[0]
            if num_points <= 1:
                continue

            optimizer.zero_grad()
            mask = torch.rand_like(seq_x[..., :1], device=device) > 0.25  # 25 % dropout
            seq_x = seq_x * mask
            # model input: (Batch=P, Channels=2, SeqLen=F)
            x_permuted = seq_x.permute(0, 2, 1)  # (P, 2, F)
            f = model.feature_extractor(x_permuted)
            loss = L_InfoNCE(f, seq_labels)
            loss.backward()
            optimizer.step()
            epoch_loss_stage1 += loss.item()
            num_seq_processed += 1

        scheduler.step()

        avg_loss = (
            epoch_loss_stage1 / num_seq_processed if num_seq_processed > 0 else 0.0
        )

        print(
            f"Pretraining Epoch {epoch + 1}/{config['pretraining_epochs']}, Avg Loss: {avg_loss:.4f}"
        )

    return model


def full_training_loop(
    config,
    model,
    train_loader,
    val_loader,
    additional_val_loader,
    device,
    optimizer,
    scheduler,
    metrics,
):
    best_model_weights = None
    best_mean_clustering_error = 1  # 1 represents 100% mean clustering error
    for epoch in tqdm(range(config["full_epochs"]), desc="Training Full Model"):
        model.train()
        model = model.to(device)
        epoch_loss_stage2 = 0.0
        num_seq_processed = 0
        w_info_sum = 0.0
        w_res_sum = 0.0
        w_feat_sum = 0.0
        w_ortho_sum = 0.0
        for batch_data in train_loader:
            seq_x = batch_data["trajectories"].to(device).squeeze(0)  # (P, F, 2)
            seq_labels = batch_data["labels"].to(device).squeeze(0)  # (P,)
            seq_t = batch_data["times"].to(device).squeeze(0)  # (P, F)
            num_points = seq_x.shape[0]
            if (
                config["augmentation_max_shift_amount"] != 0.0
                or config["augmentation_shift_percent"] != 0.0
                or config["augmentation_occlusion_percent"] != 0
            ):
                seq_x = augment_normalized_data(
                    seq_x=seq_x,
                    max_shift_amount=config["augmentation_max_shift_amount"],
                    shift_percent=config["augmentation_shift_percent"],
                    occlusion_percent=config["augmentation_occlusion_percent"],
                )

            optimizer.zero_grad()
            f, B, h_t = model(seq_x, seq_t)

            B_flat = B.view(num_points, -1)  # (P, 2F*rank)
            v = torch.cat((f, B_flat), dim=1)
            v_norm = F.normalize(v, p=2, dim=1)

            if config["use_sequence_randomization"]:
                seq_x_train = randomize_sequences_for_class(
                    seq_x=seq_x, seq_labels=seq_labels, epoch=epoch, device=device
                )
            else:
                seq_x_train = seq_x

            x_reconstructed = reconstruct_x(seq_x_train, B)  # (P, F, 2)
            x_reconstructed_permuted = x_reconstructed.permute(0, 2, 1)

            loss_residual = L_Residual(
                x_original=seq_x_train, x_reconstructed=x_reconstructed
            )

            f_reconstructed = model.feature_extractor(x_reconstructed_permuted)
            loss_infoNCE = L_InfoNCE(v_norm, seq_labels)

            if config["include_ortho_loss"]:
                loss_ortho = L_orthogonal(h_t)
            else:
                loss_ortho = torch.tensor(0.0, device=device)

            if config["include_feat_loss"]:
                loss_featdiff = L_FeatDiff(
                    f_original=f, f_reconstructed=f_reconstructed
                )
            else:
                loss_featdiff = torch.tensor(0.0, device=device)

            w_info = config["w_info"]
            w_res = config["w_res"]
            w_feat = config["w_feat"]
            w_ortho = config["w_ortho"]

            w_info_sum += w_info * loss_infoNCE
            w_res_sum += w_res * loss_residual
            w_feat_sum += w_feat * loss_featdiff
            w_ortho_sum += w_ortho * loss_ortho

            total_loss = (
                w_info * loss_infoNCE
                + w_res * loss_residual
                + w_feat * loss_featdiff
                + w_ortho * loss_ortho
            )
            total_loss.backward()
            optimizer.step()
            epoch_loss_stage2 += total_loss.item()
            num_seq_processed += 1

        scheduler.step()

        metrics["total_loss"] = epoch_loss_stage2 / num_seq_processed
        metrics["infonce_loss"] = w_info_sum / num_seq_processed
        metrics["residual_loss"] = w_res_sum / num_seq_processed
        metrics["feat_diff_loss"] = w_feat_sum / num_seq_processed
        metrics["ortho_loss"] = w_ortho_sum / num_seq_processed
        metrics["mean_clustering_error"] = evaluate_model_performance(model, val_loader)
        if additional_val_loader:
            metrics[
                f"{str.lower(config['additional_val_data'])}_mean_clustering_error"
            ] = evaluate_model_performance(model, additional_val_loader)

        if metrics["mean_clustering_error"] < best_mean_clustering_error:
            best_model_weights = copy.deepcopy(model.state_dict())

        wandb.log(metrics, step=epoch + 1)

        print(
            f"Full Training Epoch {epoch + 1}/{config['full_epochs']}, Avg Loss: {metrics['total_loss']:.4f}"
        )
        print(
            f"Epoch {epoch + 1}/{config['full_epochs']}: InfoNCE Loss: {metrics['infonce_loss']:.4f}, Res Loss: {metrics['residual_loss']:.4f}, Feat Loss: {metrics['feat_diff_loss']:.4f}, Ortho Loss: {metrics['ortho_loss']:.4f}"
        )

    model.load_state_dict(best_model_weights)
    return model


def reconstruct_x(x_original, B_estimated):
    with torch.no_grad():
        try:
            batch_size, seq_len, _ = x_original.shape
            x_flattend = x_original.reshape(batch_size, 2 * seq_len, 1)
            solution = torch.linalg.lstsq(B_estimated, x_flattend)
            c = solution.solution

            x_reconst_flat = torch.bmm(B_estimated, c)
            x_reconst = x_reconst_flat.reshape(batch_size, seq_len, 2)
        except Exception as e:
            print(f"Error occurred in x reconstruction: {e}")
    return x_reconst


def randomize_sequences_for_class(seq_x, seq_labels, epoch, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(42 + epoch)

    seq_x_train = seq_x.clone()

    if torch.rand(1, generator=generator, device=device) > 0.5:
        unique_labels = torch.unique(seq_labels)
        for label in unique_labels:
            class_indices = seq_labels == label
            if class_indices.sum() > 1:
                seq_class = seq_x[class_indices]
                num_seq_class = seq_class.size(0)
                idx = torch.randperm(num_seq_class, generator=generator, device=device)
                seq_class_shuffled = seq_class[idx]
                seq_x_train[class_indices] = seq_class_shuffled

    return seq_x_train


def load_dataset(dataset_name):
    if dataset_name == "Hopkins155":
        return Hopkins155()
    elif dataset_name == "KT3DMoSeg":
        return KT3DMoSeg()
    elif dataset_name == "Hopkins12":
        return Hopkins12()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_val_loaders(dataset, config):
    if config["strict_sequence_train_val_split"]:
        full_sequences = []
        sequences_by_name = {}
        
        for i in range(len(dataset)):
            seq = dataset[i]
            seq_name = seq["seq_name"]
            seq_type = seq["seq_type"]
            
            if seq_name not in sequences_by_name:
                sequences_by_name[seq_name] = []
            sequences_by_name[seq_name].append(i)
            
            if seq_type == "full":
                full_sequences.append(seq_name)
        
        train_seq_names, val_seq_names = train_test_split(
            full_sequences, 
            test_size=config["validation_split"], 
            random_state=42
        )
        
        train_indices = []
        for seq_name in train_seq_names:
            if config["include_partial_sequences_train"]:
                train_indices.extend(sequences_by_name[seq_name])
            else:
                for idx in sequences_by_name[seq_name]:
                    seq = dataset[idx]
                    if seq["seq_type"] == "full":
                        train_indices.append(idx)
        
        val_indices = []
        for seq_name in val_seq_names:
            if config["include_partial_sequences_val"]:
                val_indices.extend(sequences_by_name[seq_name])
            else:
                for idx in sequences_by_name[seq_name]:
                    seq = dataset[idx]
                    if seq["seq_type"] == "full":
                        val_indices.append(idx)
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
    else:
        train_dataset, val_dataset = train_test_split(
            range(len(dataset)), 
            test_size=config["validation_split"], 
            random_state=42
        )
        train_dataset = Subset(dataset, train_dataset)
        val_dataset = Subset(dataset, val_dataset)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False
    )
    
    return train_loader, val_loader