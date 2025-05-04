import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from training import Hopkins155
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np


def load_model():    
    model = TrajectoryEmbeddingModel()
    load_path = 'out/models/trained_model_weights_normalized4.pt'

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(load_path, map_location=target_device)
    
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(target_device)
    model.eval()
    return model

def calculate_clustering_error(labels_true, labels_pred):
    """Hungarian algorithm for best matching"""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    cm = contingency_matrix(labels_true, labels_pred)

    row_ind, col_ind = linear_sum_assignment(-cm)
    correctly_assigned_count = cm[row_ind, col_ind].sum()
    total_points = np.sum(cm)
    if total_points == 0:
        return 0.0
    accuracy = correctly_assigned_count / total_points
    error_rate = 1.0 - accuracy
    return error_rate

def load_trajectory_data():
    dataset = Hopkins155()
    loaded_data = DataLoader(dataset, batch_size=1)
    return loaded_data

def evaluate_model_performance(model):
    data = load_trajectory_data()
    individual_error_rates = []
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    with torch.no_grad():
        for sequence in data:
            seq_x = sequence['trajectories'].to(target_device).squeeze(0)
            seq_labels_gt = sequence['labels'].squeeze(0)
            k = sequence['num_clusters'].item()
            seq_name = sequence['name'][0]
            
            seq_x_permuted = seq_x.permute(0, 2, 1)
            f = model.feature_extractor(seq_x_permuted)
            f = f.cpu().numpy()
            
            k = sequence['num_clusters'].item()
            clusters = AgglomerativeClustering(n_clusters=k, linkage='ward', compute_distances=False)
            predicted_labels = clusters.fit_predict(f)
            error_rate = calculate_clustering_error(seq_labels_gt.numpy(), predicted_labels)
            individual_error_rates.append(error_rate)

    mean_error_rate = sum(individual_error_rates) / len(individual_error_rates)
    median_error_rate = np.median(individual_error_rates)
    print(f"\nEvaluation Complete.")
    print(f"Mean Clustering Error: {mean_error_rate * 100:.2f}%")
    print(f"Median Clustering Error: {median_error_rate * 100:.2f}%")

    return mean_error_rate

def main():
    model = load_model()
    error_rate = evaluate_model_performance(model)

if __name__ == '__main__':
    main()