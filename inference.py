import os

import torch
import torch.nn.functional as F
from models.trajectory_embedder import TrajectoryEmbeddingModel
from datasets import load_dataset, get_train_val_loaders
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


def load_model(load_path="out/models/hopk155_100_200_split_incfeat_ortho_alph0.pt"):
    model = TrajectoryEmbeddingModel()

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


def cluster_unprocessed_trajectories(data, cluster_algo_name="hierarchical"):
    individual_error_rates = []

    nmi_scores = []
    ari_scores = []
    for sequence in data:
        seq = sequence["trajectories"].squeeze(0)

        seq_labels_gt = sequence["labels"].squeeze(0)
        k_field = sequence["num_clusters"]
        k = k_field.item() if torch.is_tensor(k_field) else int(k_field)
        seq = seq.reshape(seq.shape[0], -1)

        predicted_labels = None
        if cluster_algo_name == "hierarchical":
            clusters = AgglomerativeClustering(
                n_clusters=k, linkage="ward", compute_distances=False
            )
            predicted_labels = clusters.fit_predict(seq)
        elif cluster_algo_name == "kmeans":
            clusters = KMeans(n_clusters=k, random_state=0, n_init=10)
            predicted_labels = clusters.fit_predict(seq)
        elif cluster_algo_name == "spectral":
            clusters = SpectralClustering(
                n_clusters=k, random_state=0, affinity="rbf", n_neighbors=20
            )
            predicted_labels = clusters.fit_predict(seq)
        else:
            print(f"Error: Unknown clustering algorithm '{cluster_algo_name}'")
            continue

        nmi = normalized_mutual_info_score(seq_labels_gt, predicted_labels)
        ari = adjusted_mutual_info_score(seq_labels_gt, predicted_labels)
        nmi_scores.append(nmi)
        ari_scores.append(ari)

        error_rate = calculate_clustering_error(seq_labels_gt.numpy(), predicted_labels)
        individual_error_rates.append(error_rate)

    mean_error_rate = sum(individual_error_rates) / len(individual_error_rates)
    median_error_rate = np.median(individual_error_rates)
    mean_nmi = sum(nmi_scores) / len(nmi_scores)
    mean_ari = sum(ari_scores) / len(ari_scores)
    print(f"Evaluation Complete with {cluster_algo_name} clustering:")
    print(f"Mean Clustering Error: {mean_error_rate * 100:.2f}%")
    print(f"Median Clustering Error: {median_error_rate * 100:.2f}%")
    print(f"Mean NMI Score: {mean_nmi:.3f}")
    print(f"Mean ARI Score: {mean_ari:.3f}\n")

    return mean_error_rate


def compare_all_clustering_methods_unprocessed():
    data = load_dataset("hopkins155")
    algorithms = ["hierarchical", "kmeans", "spectral"]
    for algorithm in algorithms:
        cluster_unprocessed_trajectories(data, algorithm)


def evaluate_model_performance(
    model,
    data,
    dataset_name,
    generate_video=False,
    cluster_algo_name="hierarchical",
    model_name=None,
    device_str="cuda",
):
    individual_error_rates = []
    if isinstance(device_str, torch.device):
        target_device = device_str
    else:
        target_device = torch.device(device_str)
    nmi_scores = []
    ari_scores = []
    model.to(target_device)
    model.eval()
    with torch.no_grad():
        for sequence in data:
            seq_x = sequence["trajectories"].to(target_device).squeeze(0)
            seq_t = sequence["times"].to(target_device).squeeze(0)

            seq_labels_gt = sequence["labels"].squeeze(0)
            k_field = sequence["num_clusters"]
            k = k_field.item() if torch.is_tensor(k_field) else int(k_field)
            seq_name = sequence["seq_name"][0]

            f, B, _ = model(seq_x, seq_t)
            B_flat = B.view(B.size(0), -1)
            f_norm = F.normalize(f, p=2, dim=1)
            B_flat_norm = F.normalize(B_flat, p=2, dim=1)
            v = torch.cat((f_norm, B_flat_norm), dim=1).cpu().numpy()
            f = f.cpu().numpy()

            predicted_labels = None
            if cluster_algo_name == "hierarchical":
                clusters = AgglomerativeClustering(
                    n_clusters=k, linkage="ward", compute_distances=False
                )
                predicted_labels = clusters.fit_predict(v)
            elif cluster_algo_name == "kmeans":
                clusters = KMeans(n_clusters=k, random_state=0, n_init=10)
                predicted_labels = clusters.fit_predict(v)
            elif cluster_algo_name == "spectral":
                # tested options: 'rbf', 'nearest_neighbor'; possibly worth experminenting with different hyp. params here
                clusters = SpectralClustering(
                    n_clusters=k, random_state=0, affinity="rbf", n_neighbors=20
                )
                predicted_labels = clusters.fit_predict(v)
            else:
                print(f"Error: Unknown clustering algorithm '{cluster_algo_name}'")
                continue

            if generate_video:
                seq_unnormalized = sequence["unnormalized_trajectories"]
                generate_cluster_video(
                    labels_pred=predicted_labels,
                    labels_true=seq_labels_gt,
                    sequences_unnormalized=seq_unnormalized,
                    dataset_name=dataset_name,
                    seq_name=seq_name,
                    model_name=model_name,
                )

            nmi = normalized_mutual_info_score(seq_labels_gt, predicted_labels)
            ari = adjusted_mutual_info_score(seq_labels_gt, predicted_labels)
            nmi_scores.append(nmi)
            ari_scores.append(ari)

            error_rate = calculate_clustering_error(
                seq_labels_gt.numpy(), predicted_labels
            )
            individual_error_rates.append(error_rate)

    mean_error_rate = sum(individual_error_rates) / len(individual_error_rates)
    median_error_rate = np.median(individual_error_rates)
    mean_nmi = sum(nmi_scores) / len(nmi_scores)
    mean_ari = sum(ari_scores) / len(ari_scores)
    print(f"Evaluation Complete with {cluster_algo_name} clustering:")
    print(f"Mean Clustering Error: {mean_error_rate * 100:.2f}%")
    print(f"Median Clustering Error: {median_error_rate * 100:.2f}%")
    print(f"Mean NMI Score: {mean_nmi:.3f}")
    print(f"Mean ARI Score: {mean_ari:.3f}\n")

    return mean_error_rate


def compare_all_clustering_methods(model, data):
    algorithms = ["hierarchical", "kmeans", "spectral"]
    for algorithm in algorithms:
        evaluate_model_performance(model, data, algorithm)


def generate_cluster_video(
    labels_pred, labels_true, sequences_unnormalized, dataset_name, seq_name, model_name
):
    in_video_path = f"data/{dataset_name}/{seq_name}/{seq_name}.avi"
    out_video_directory = f"data/{dataset_name}/{seq_name}/out/"
    if not os.path.exists(out_video_directory):
        os.makedirs(out_video_directory)

    if not model_name:
        model_name = "video_out"
    out_video_path = f"{out_video_directory}/{model_name}_{seq_name}.avi"

    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0

    cluster_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]

    labels_true = np.array(labels_true)
    unique_true_labels = np.unique(labels_true)
    unique_pred_labels = np.unique(labels_pred)

    cm = contingency_matrix(labels_true, labels_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)

    max_pred_label = np.max(unique_pred_labels)
    lookup_table = np.full(max_pred_label + 1, fill_value=-1, dtype=int)

    for true_row_idx, pred_col_idx in zip(row_ind, col_ind):
        original_pred_label = unique_pred_labels[pred_col_idx]
        target_true_label = unique_true_labels[true_row_idx]

        lookup_table[original_pred_label] = target_true_label

    labels_pred_new = lookup_table[labels_pred]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number >= sequences_unnormalized.shape[2]:
            out.write(frame)
            frame_number += 1
            continue

        points_to_draw = sequences_unnormalized[0].permute(1, 0, 2)[frame_number]

        for i, point in enumerate(points_to_draw):
            x, y = int(point[0]), int(point[1])

            label_pred = labels_pred_new[i]
            label_true = labels_true[i]

            is_correct = label_pred == label_true

            color = cluster_colors[label_pred]

            if is_correct:
                cv2.circle(frame, (x, y), radius=3, color=color, thickness=-1)
            else:
                line_length = 3
                cv2.line(
                    frame,
                    (x - line_length, y - line_length),
                    (x + line_length, y + line_length),
                    color=color,
                    thickness=1,
                )
                cv2.line(
                    frame,
                    (x - line_length, y + line_length),
                    (x + line_length, y - line_length),
                    color=color,
                    thickness=1,
                )

        out.write(frame)
        frame_number += 1

    print(f"Processing complete. Video saved to {out_video_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def perform_inference(model_name, config):
    dataset_name = config["train_data"]
    load_path = f"out/models/{model_name}.pth"
    dataset = load_dataset(dataset_name=dataset_name)
    train_loader, val_loader = get_train_val_loaders(
        dataset=dataset,
        strict_sequence_train_val_split=config["strict_sequence_train_val_split"],
        val_split_size=config["validation_split"],
        include_partial_sequences_train=config["include_partial_sequences_train"],
        include_partial_sequences_val="include_partial_sequences_val",
    )
    model = load_model(load_path=load_path)
    error_rate = evaluate_model_performance(
        model,
        val_loader,
        dataset_name=dataset_name,
        generate_video=True,
        model_name=model_name,
    )
