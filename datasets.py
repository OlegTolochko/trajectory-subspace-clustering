import torch
import os
import re
import scipy
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random


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

                if "y" in mat_data:
                    y_data_load = mat_data["y"]

                coords_2PF = x_data_load[0:2, :, :]  # (2, P, F)
                num_points = coords_2PF.shape[1]
                num_frames = coords_2PF.shape[2]
                trajectories = np.transpose(coords_2PF, (1, 2, 0))  # (P, F, 2)
                base_time = torch.arange(num_frames, dtype=torch.float32) / (
                    num_frames - 1
                )
                time_vectors = base_time.expand(num_points, -1)

                y_coords_2PF = y_data_load[0:2, :, :]  # (2, P, F)
                y_trajectories = np.transpose(y_coords_2PF, (1, 2, 0))  # (P, F, 2)

                if "s" in mat_data:
                    labels_load = mat_data["s"].reshape(-1)

                pattern = r"_g(.+)$"
                match = re.search(pattern, seq_name)

                if match:
                    seq_type = match.group(1)
                    seq_name = seq_name.rsplit("_g", 1)[0]
                else:
                    seq_type = "full"

                self.sequence_data.append(
                    {
                        "seq_name": seq_name,
                        "seq_type": seq_type,
                        "trajectories": trajectories.astype(np.float32),
                        "unnormalized_trajectories": y_trajectories.astype(np.float32),
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
        seq_type = seq_info["seq_type"]
        y_trajectories = seq_info["unnormalized_trajectories"]

        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        time_tensor = time_vectors.long()
        num_clusters = len(torch.unique(labels_tensor))

        return {
            "trajectories": trajectories_tensor,
            "unnormalized_trajectories": y_trajectories,
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
                        "seq_name": seq_name,
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

        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        time_tensor = time_vectors.long()
        num_clusters = len(torch.unique(labels_tensor))

        return {
            "trajectories": trajectories_tensor,
            "labels": labels_tensor,
            "times": time_tensor,
            "seq_name": seq_name,
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

                    num_points = coords.shape[1]
                    num_frames = coords.shape[2]
                    coords = np.transpose(coords, (1, 2, 0))  # (P, F, 2)

                    x_coords = coords[:, :, 0]
                    y_coords = coords[:, :, 1]

                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)

                    x_range = x_max - x_min if x_max != x_min else 1.0
                    y_range = y_max - y_min if y_max != y_min else 1.0

                    coords[:, :, 0] = (x_coords - x_min) / x_range
                    coords[:, :, 1] = (y_coords - y_min) / y_range

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


def load_dataset(dataset_name):
    if str.lower(dataset_name) == "hopkins155":
        return Hopkins155()
    elif str.lower(dataset_name) == "kt3dmoseg":
        return KT3DMoSeg()
    elif str.lower(dataset_name) == "hopkins12":
        return Hopkins12()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_val_loaders(
    dataset,
    val_split_size,
    strict_sequence_train_val_split,
    include_partial_sequences_train,
    include_partial_sequences_val,
    random_state: int | None = 42,
):
    if val_split_size == 1.0:
        train_loader = DataLoader(dataset, shuffle=True)
        val_loader = DataLoader(dataset, shuffle=False)
        return train_loader, val_loader

    if strict_sequence_train_val_split:
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
            full_sequences, test_size=val_split_size, random_state=random_state
        )

        train_indices = []
        for seq_name in train_seq_names:
            if include_partial_sequences_train:
                train_indices.extend(sequences_by_name[seq_name])
            else:
                for idx in sequences_by_name[seq_name]:
                    seq = dataset[idx]
                    if seq["seq_type"] == "full":
                        train_indices.append(idx)

        val_indices = []
        for seq_name in val_seq_names:
            if include_partial_sequences_val:
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
            range(len(dataset)), test_size=val_split_size, random_state=random_state
        )
        train_dataset = Subset(dataset, train_dataset)
        val_dataset = Subset(dataset, val_dataset)

    train_loader = DataLoader(train_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, shuffle=False)

    return train_loader, val_loader
