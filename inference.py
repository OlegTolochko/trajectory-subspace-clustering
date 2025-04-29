import torch
from models.trajectory_embedder import TrajectoryEmbeddingModel
from training import Hopkins155
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


def load_model():    
    model = TrajectoryEmbeddingModel()
    load_path = 'out/models/trained_model_weights.pt'

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(load_path, map_location=target_device)
    
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    model.to(target_device)
    model.eval()
    return model

def calculate_clustering_error(ground_truth, predicted):
    num_trajectories = len(ground_truth)
    correct_count = 0
    for gt_label, p_label in zip(ground_truth, predicted):
        if gt_label == p_label:
            correct_count += 1
    
    return 1 - correct_count/num_trajectories

def load_trajectory_data():
    dataset = Hopkins155()
    loaded_data = DataLoader(dataset, batch_size=1)
    return loaded_data

def evaluate_model_performance(model):
    data = load_trajectory_data()
    individual_error_rates = []
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for sequence in data:
            seq_x = sequence['trajectories'].to(target_device).squeeze(0)
            seq_labels_gt = sequence['labels'].squeeze(0)
            k = sequence['num_clusters'].item()
            
            seq_x_permuted = seq_x.permute(0, 2, 1)
            f = model.feature_extractor(seq_x_permuted)
            f = f.cpu().numpy()
            
            k = sequence['num_clusters'].item()
            clusters = AgglomerativeClustering(n_clusters=k)
            predicted_labels = clusters.fit_predict(f)
            error_rate = calculate_clustering_error(seq_labels_gt.numpy(), predicted_labels)

            individual_error_rates.append(error_rate)
        
    mean_error_rate = sum(individual_error_rates)/len(individual_error_rates)
    return mean_error_rate

def main():
    model = load_model()
    error_rate = evaluate_model_performance(model)
    print(f"Error rate for Hopkins155 dataset: {error_rate}")

if __name__ == '__main__':
    main()