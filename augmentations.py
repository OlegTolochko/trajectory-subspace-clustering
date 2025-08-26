import torch
import random
import numpy as np

def trajectory_point_shifting(seq_x, max_shift_amount=0.1, shift_percent=0.1):
    seq_x_shifted = seq_x.clone()

    num_points = seq_x.shape[0]

    shift_mask = torch.rand(num_points, device=seq_x.device) < shift_percent
    num_points_to_shift = torch.sum(shift_mask).item()

    if num_points_to_shift > 0:
        shifts = torch.empty(
            (num_points_to_shift, seq_x.shape[1], seq_x.shape[2]),
            device=seq_x.device,
            dtype=seq_x.dtype,
        )
        shifts.uniform_(-max_shift_amount, max_shift_amount)

        seq_x_shifted[shift_mask] += shifts

        seq_x_shifted = torch.clamp(seq_x_shifted, 0, 1)

    return seq_x_shifted


def individual_point_shifting(seq_x, max_shift_amount=0.1, shift_percent=0.1):
    seq_x_shifted = seq_x.clone()
    
    shift_mask = torch.rand((seq_x.shape[0], seq_x.shape[1]), device=seq_x.device) < shift_percent
    
    shifts = torch.empty_like(seq_x)
    shifts.uniform_(-max_shift_amount, max_shift_amount)
    
    shift_mask_expanded = shift_mask.unsqueeze(-1).expand(-1, -1, 2)
    seq_x_shifted[shift_mask_expanded] += shifts[shift_mask_expanded]
    
    seq_x_shifted = torch.clamp(seq_x_shifted, 0, 1)
    
    return seq_x_shifted


def occlude_seq(seq_x, occlusion_percent):
    seq_x_occluded = seq_x.clone()
    num_points = seq_x.shape[0]
    device = seq_x.device

    occlusion_mask = torch.rand(num_points, device=device) < occlusion_percent

    i = 0
    while i < num_points:
        if not occlusion_mask[i]:
            i += 1
            continue
        start_idx = i

        while i < num_points and occlusion_mask[i]:
            i += 1
        end_idx = i - 1
        source_idx = start_idx - 1

        if start_idx == 0:
            if end_idx == num_points - 1:
                continue
            source_idx = end_idx + 1

        replacement_tensor = seq_x[source_idx]
        seq_x_occluded[start_idx : end_idx + 1] = replacement_tensor

    return seq_x_occluded

def trajectory_shifting(seq, max_shift_amount, shift_percent):
    seq_x_shifted = seq.clone()

    num_points = seq.shape[0]

    shift_mask = torch.rand(num_points, device=seq.device) < shift_percent
    num_points_to_shift = torch.sum(shift_mask).item()

    if num_points_to_shift > 0:
        shifts = torch.empty(
            (num_points_to_shift, 2),
            device=seq.device,
            dtype=seq.dtype,
        )
        shifts.uniform_(-max_shift_amount, max_shift_amount)

        seq_x_shifted[shift_mask] += torch.unsqueeze(shifts, dim=1)

        seq_x_shifted = torch.clamp(seq_x_shifted, 0, 1)

    return seq_x_shifted


def sequence_shifting(seq, max_shift_amount=0.3):
    seq_shifted = seq.clone()

    shift_y = random.uniform(-max_shift_amount, max_shift_amount)
    shift_x = random.uniform(-max_shift_amount, max_shift_amount)
    seq_shifted[:, :, 0] += shift_x
    seq_shifted[:, :, 1] += shift_y
    seq_shifted = torch.clamp(seq_shifted, 0, 1)

    return seq_shifted


def occlude_seq_chunkwise(seq, occlusion_percent, max_num_chunks):
    seq_occluded = seq.clone()
    num_points = seq.shape[0]
    device = seq.device

    occlusion_mask = torch.zeros(num_points, dtype=torch.bool, device=device)

    occlusion_positions = torch.randint(0, num_points, (max_num_chunks,), device=device)

    max_length_per_chunk = int(
        num_points * (occlusion_percent * 2 / max_num_chunks + 0.01)
    )
    if max_length_per_chunk == 0:
        max_length_per_chunk = 1

    chunk_lengths = torch.randint(
        1, max_length_per_chunk + 1, (max_num_chunks,), device=device
    )

    for i in range(max_num_chunks):
        start = occlusion_positions[i].item()
        length = chunk_lengths[i].item()
        end = min(start + length, num_points)
        occlusion_mask[start:end] = True

    i = 0
    while i < num_points:
        if not occlusion_mask[i]:
            i += 1
            continue
        start_idx = i

        while i < num_points and occlusion_mask[i]:
            i += 1
        end_idx = i - 1
        source_idx = start_idx - 1

        if start_idx == 0:
            if end_idx == num_points - 1:
                continue
            source_idx = end_idx + 1

        replacement_tensor = seq[source_idx]
        seq_occluded[start_idx : end_idx + 1] = replacement_tensor

    return seq_occluded


def randomly_augment_seq(seq, config):
    seq_augmented = seq
    # Individual Point Shifting:
    if config["augmentation_individual_shift_percent"] > 0:
        individual_shift_percentages = np.arange(
            0, config["augmentation_individual_shift_percent"], 0.05
        )
        max_individual_shift_amounts = np.arange(
            0, config["augmentation_individual_max_shift_amount"], 0.05
        )
        individual_shift_percent = random.choice(individual_shift_percentages)
        max_individual_shift_amount = random.choice(max_individual_shift_amounts)

        seq_augmented = individual_point_shifting(
            seq_x=seq_augmented,
            max_shift_amount=max_individual_shift_amount,
            shift_percent=individual_shift_percent,
        )
        
    # Trajectory-wise Point Shifting:
    if config["augmentation_trajectory_individual_shift_percent"] > 0:
        individual_shift_percentages = np.arange(
            0, config["augmentation_trajectory_individual_shift_percent"], 0.05
        )
        max_individual_shift_amounts = np.arange(
            0, config["augmentation_trajectory_individual_shift_distance"], 0.05
        )
        individual_shift_percent = random.choice(individual_shift_percentages)
        max_individual_shift_amount = random.choice(max_individual_shift_amounts)

        seq_augmented = trajectory_point_shifting(
            seq_x=seq_augmented,
            max_shift_amount=max_individual_shift_amount,
            shift_percent=individual_shift_percent,
        )
    
    # Trajectory-wise Shifting:
    if config["augmentation_trajectory_shift_percent"] > 0:
        individual_shift_percentages = np.arange(
            0, config["augmentation_trajectory_shift_percent"], 0.05
        )
        max_individual_shift_amounts = np.arange(
            0, config["augmentation_trajectory_max_shift_amount"], 0.05
        )
        individual_shift_percent = random.choice(individual_shift_percentages)
        max_individual_shift_amount = random.choice(max_individual_shift_amounts)

        seq_augmented = trajectory_shifting(
            seq=seq_augmented,
            max_shift_amount=max_individual_shift_amount,
            shift_percent=individual_shift_percent,
        )

    # Full Sequence Shifting:
    if config["augmentation_full_max_shift_amount"] > 0:
        max_full_shift_amounts = np.arange(
            0, config["augmentation_full_max_shift_amount"], 0.05
        )
        max_full_shift_amount = random.choice(max_full_shift_amounts)

        seq_augmented = sequence_shifting(
            seq=seq_augmented, max_shift_amount=max_full_shift_amount
        )

    # Chunkwise Occlusion:
    if config["augmentation_chunkwise_occlusion_percent"] > 0:
        chunkwise_occlusion_percentages = np.arange(
            0, config["augmentation_chunkwise_occlusion_percent"], 0.05
        )
        chunkwise_occlusion_max_chunk_amounts = np.arange(
            1, config["augmentation_chunkwise_occlusion_max_chunk_amount"] + 1, 1
        )
        chunkwise_occlusion_percent = random.choice(chunkwise_occlusion_percentages)
        chunkwise_occlusion_max_num_chunks = random.choice(
            chunkwise_occlusion_max_chunk_amounts
        )

        seq_augmented = occlude_seq_chunkwise(
            seq_augmented,
            occlusion_percent=chunkwise_occlusion_percent,
            max_num_chunks=chunkwise_occlusion_max_num_chunks,
        )

    # Point-wise Occlusion:
    if config["augmentation_occlusion_percent"] > 0:
        occlusion_percentages = np.arange(
            0, config["augmentation_occlusion_percent"], 0.05
        )
        occlusion_percent = random.choice(occlusion_percentages)

        seq_augmented = occlude_seq(seq_augmented, occlusion_percent=occlusion_percent)

    return seq_augmented