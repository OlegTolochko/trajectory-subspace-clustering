import torch
import os
import re
import scipy
from torch.utils.data import Dataset
import numpy as np


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
                if "x" in mat_data:
                    x_data_load = mat_data["x"]

                coords_2PF = x_data_load[0:2, :, :]  # (2, P, F)
                num_points = coords_2PF.shape[1]
                num_frames = coords_2PF.shape[2]
                trajectories = np.transpose(coords_2PF, (1, 2, 0))  # (P, F, 2)
                base_time = torch.arange(num_frames)
                time_vectors = base_time.expand(num_points, -1)

                if "s" in mat_data:
                    labels_load = mat_data["s"].reshape(-1)
                    
                pattern = r'_g(.+)$'
                match = re.search(pattern, seq_name)
                
                if match:
                    seq_type = match.group(1)
                    seq_name = seq_name.rsplit('_g', 1)[0]
                else:
                    seq_type = "full"
                
                self.sequence_data.append(
                    {
                        "seq_name": seq_name,
                        "seq_type": seq_type,
                        "trajectories": trajectories.astype(np.float32),
                        "times": time_vectors,
                        "labels": labels_load.astype(np.int64),
                    }
                )

            except Exception as e:
                print(f"Error loading or processing {mat_file_path}: {e}")

        print(f"finished loading data for {len(self.sequence_data)} sequences")

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        if idx >= len(self.sequence_data):
            raise IndexError("Index out of bounds")

        seq_info = self.sequence_data[idx]
        trajectories = seq_info["trajectories"]
        labels = seq_info["labels"]
        seq_name = seq_info["seq_name"]
        time_vectors = seq_info["times"]
        seq_type = seq_type["seq_type"]

        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        time_tensor = time_vectors.long()
        num_clusters = len(torch.unique(labels_tensor))

        return {
            "trajectories": trajectories_tensor,
            "labels": labels_tensor,
            "times": time_tensor,
            "seq_name": seq_name,
            "seq_type": seq_type,
            "num_clusters": num_clusters,
        }


class Hopkins12(Dataset):
    def __init__(self, root_dir="data/Hopkins12/"):
        self.root_dir = root_dir
        self.sequence_data = []

        print(f"Loading Hopkins12 data from: {root_dir}")
        for seq_name in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq_name)
            if os.path.isdir(seq_path):
                mat_file_name = f"{seq_name}_truth.mat"
                mat_file_path = os.path.join(seq_path, mat_file_name)

            try:
                mat_data = scipy.io.loadmat(mat_file_path)
                x_data_load = None
                if "x" in mat_data:
                    x_data_load = mat_data["x"]

                coords_2PF = x_data_load[0:2, :, :]  # (2, P, F)
                num_points = coords_2PF.shape[1]
                num_frames = coords_2PF.shape[2]
                trajectories = np.transpose(coords_2PF, (1, 2, 0))  # (P, F, 2)
                base_time = torch.arange(num_frames)
                time_vectors = base_time.expand(num_points, -1)

                if "s" in mat_data:
                    labels_load = mat_data["s"].reshape(-1)

                self.sequence_data.append(
                    {
                        "name": seq_name,
                        "trajectories": trajectories.astype(np.float32),
                        "times": time_vectors,
                        "labels": labels_load.astype(np.int64),
                    }
                )

            except Exception as e:
                print(f"Error loading or processing {mat_file_path}: {e}")

        print(f"finished loading data for {len(self.sequence_data)} sequences")

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        if idx >= len(self.sequence_data):
            raise IndexError("Index out of bounds")

        seq_info = self.sequence_data[idx]
        trajectories = seq_info["trajectories"]
        labels = seq_info["labels"]
        seq_name = seq_info["name"]
        time_vectors = seq_info["times"]

        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        time_tensor = time_vectors.long()
        num_clusters = len(torch.unique(labels_tensor))

        return {
            "trajectories": trajectories_tensor,
            "labels": labels_tensor,
            "times": time_tensor,
            "name": seq_name,
            "num_clusters": num_clusters,
        }


class KT3DMoSeg(Dataset):
    def __init__(self, root_dir="data/KT3DMoSeg/"):
        self.root_dir = root_dir
        self.sequence_data = []

        print(f"Loading KT3DMoSeg data from: {root_dir}")
        loaded_file_count = 0

        for mat_file in sorted(os.listdir(root_dir)):
            if mat_file.endswith(".mat"):
                mat_file_path = os.path.join(root_dir, mat_file)
                try:
                    loaded_mat = scipy.io.loadmat(mat_file_path)
                    mat_data_struct = loaded_mat["Data"]

                    trajectories_load = None
                    labels_load = None

                    current_data_fields = mat_data_struct.dtype.names

                    if "ySparse" in current_data_fields:
                        trajectories_load = mat_data_struct[0, 0]["ySparse"]

                    if "GtLabel" in current_data_fields:
                        labels_load = mat_data_struct[0, 0]["GtLabel"].reshape(-1)

                    coords = trajectories_load[0:2]
                    scale = float(np.hypot(1242, 375))
                    coords = (coords - 0.5 * scale) / (0.5 * scale)

                    num_points = coords.shape[1]
                    num_frames = coords.shape[2]
                    coords = np.transpose(coords, (1, 2, 0))  # (P, F, 2)

                    base_time = torch.arange(num_frames)
                    time_vectors = base_time.unsqueeze(0).expand(num_points, -1)

                    self.sequence_data.append(
                        {
                            "name": mat_file,
                            "trajectories": coords.astype(np.float32),
                            "times": time_vectors,
                            "labels": labels_load.astype(np.int64),
                        }
                    )
                    loaded_file_count += 1

                except Exception as e:
                    print(f"Error loading or processing {mat_file_path}: {e}")
                    import traceback

                    traceback.print_exc()

        print(f"Finished loading data for {len(self.sequence_data)} sequences")

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        if idx >= len(self.sequence_data):
            raise IndexError("Index out of bounds")

        seq_info = self.sequence_data[idx]
        trajectories = seq_info["trajectories"]
        labels = seq_info["labels"]
        seq_name = seq_info["name"]
        time_vectors = seq_info["times"]

        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        time_tensor = time_vectors.long()

        num_clusters = len(torch.unique(labels_tensor))

        return {
            "trajectories": trajectories_tensor,
            "labels": labels_tensor,
            "times": time_tensor,
            "name": seq_name,
            "num_clusters": num_clusters,
        }


def augment_normalized_data(
    seq_x, max_shift_amount=0.1, shift_percent=0.1, occlusion_percent=0.1
):
    seq_augmented = seq_x
    if max_shift_amount != 0 and shift_percent != 0:
        seq_augmented = shift_seq(
            seq_x=seq_x, max_shift_amount=max_shift_amount, shift_percent=shift_percent
        )

    if occlusion_percent != 0:
        seq_augmented = occlude_seq(
            seq_x=seq_augmented, occlusion_percent=occlusion_percent
        )

    return seq_augmented


def shift_seq(seq_x, max_shift_amount=0.1, shift_percent=0.1):
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
