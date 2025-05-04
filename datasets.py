import torch
import os
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