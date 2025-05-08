import torch
import sys
sys.path.append("/home/wiss/xian/Python_code/oleg/trajectory_embedding_learning/trajectory-subspace-clustering")
from models.trajectory_embedder import TrajectoryEmbeddingModel
from datasets import Hopkins155
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import os
import pickle
feature_save_path = "./out/results"
def load_model():
    model = TrajectoryEmbeddingModel()
    load_path = './out/models/trained_model_weights_normalized.pt'

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
    dataset = Hopkins155("./data/Hopkins155/")
    loaded_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loaded_data


def evaluate_model_performance(model, data, cluster_algo_name='hierarchical'):
    individual_error_rates = []
    target_device = torch.device("cpu")
    model.to(target_device)
    output_feature_dict = {}
    output_label_dict = {}
    with torch.no_grad():
        for sequence in data:
            seq_x = sequence['trajectories'].to(target_device).squeeze(0)
            seq_t = sequence['times'].squeeze(0)
            seq_labels_gt = sequence['labels'].squeeze(0)
            k_field = sequence['num_clusters']
            k = k_field.item() if torch.is_tensor(k_field) else int(k_field)
            seq_name = sequence['name'][0]

            f, B = model(seq_x, seq_t)
            B_flat = B.view(B.size(0), -1)
            v = torch.cat((f, B_flat), dim=1)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            feats_np = v.cpu().numpy()
            name = sequence["name"][0]
            if name not in output_feature_dict.keys():
                output_feature_dict[sequence["name"][0]] = feats_np
            else:
                print(f"{name} already exists")
                output_feature_dict[sequence["name"][0]] = [output_feature_dict[sequence["name"][0]]]
                output_feature_dict[sequence["name"][0]].append(feats_np)

            predicted_labels = None
            if cluster_algo_name == 'hierarchical':
                clusters = AgglomerativeClustering(n_clusters=k, linkage='ward', compute_distances=False)
                predicted_labels = clusters.fit_predict(feats_np)
            elif cluster_algo_name == 'kmeans':
                clusters = KMeans(n_clusters=k, random_state=0, n_init=10)
                predicted_labels = clusters.fit_predict(feats_np)
            elif cluster_algo_name == 'spectral':
                # tested options: 'rbf', 'nearest_neighbor'; possibly worth experminenting with different hyp. params here
                clusters = SpectralClustering(n_clusters=k, random_state=0, affinity='rbf', n_neighbors=20)
                predicted_labels = clusters.fit_predict(feats_np)
            else:
                print(f"Error: Unknown clustering algorithm '{cluster_algo_name}'")
                continue

            if name not in output_feature_dict.keys():
                output_label_dict[sequence["name"][0]] = seq_labels_gt.numpy()
            else:
                print(f"{name} already exists")
                output_label_dict[sequence["name"][0]] = [output_label_dict[sequence["name"][0]]]
                output_label_dict[sequence["name"][0]].append(seq_labels_gt.numpy())

            # output_label_dict[sequence["name"][0]] = seq_labels_gt.numpy()
            error_rate = calculate_clustering_error(seq_labels_gt.numpy(), predicted_labels)
            individual_error_rates.append(error_rate)

    mean_error_rate = sum(individual_error_rates) / len(individual_error_rates)
    median_error_rate = np.median(individual_error_rates)
    print(f"Evaluation Complete with {cluster_algo_name} clustering:")
    print(f"Mean Clustering Error: {mean_error_rate * 100:.2f}%")
    print(f"Median Clustering Error: {median_error_rate * 100:.2f}% \n")

    if not os.path.isdir(feature_save_path):
        print("Save feature folder doesn't exists")
        os.mkdir(feature_save_path)
    with open(feature_save_path + "/trajectory_embedding", 'wb') as handle:
        pickle.dump(output_feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(feature_save_path + "/trajectory_embedding_labels", 'wb') as handle:
        pickle.dump(output_label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # np.save(feature_save_path + "/trajectory_embedding.npy", output_feature_dict)
    # d2 = np.load(feature_save_path + "/trajectory_embedding.npy", allow_pickle=True)
    # print(sequence["name"][0])
    # print(d2.item().get(sequence["name"][0]))

    return mean_error_rate


def compare_all_clustering_methods(model, data):
    algorithms = ['hierarchical', 'kmeans', 'spectral']
    for algorithm in algorithms:
        evaluate_model_performance(model, data, algorithm)


def main():
    model = load_model()
    data = load_trajectory_data()
    error_rate = compare_all_clustering_methods(model, data)


if __name__ == '__main__':
    main()