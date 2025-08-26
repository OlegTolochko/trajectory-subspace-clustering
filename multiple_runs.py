import os
import copy
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Any
import json

from config import CONFIG
from training import train_model
from training_unsupervised import train_model_unsupervised
from datasets import get_train_val_loaders, load_dataset
from inference import evaluate_model_performance


def run_multiple_experiments(
    num_runs: int = 50,
    config: Dict[str, Any] = None,
    unsupervised: bool = False,
    save_results: bool = True,
    results_dir: str = "out/multiple_runs_results",
    name: str = "",
) -> Dict[str, Any]:
    if config is None:
        config = copy.deepcopy(CONFIG)

    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    all_results = []

    print(f"Starting {num_runs} runs with different random states...")

    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}")

        random_state = 42 + run_idx

        run_config = copy.deepcopy(config)
        run_config["random_state"] = random_state

        main_dataset = load_dataset(run_config["train_data"])

        main_train_loader, main_val_loader = get_train_val_loaders(
            main_dataset,
            val_split_size=run_config["validation_split"],
            strict_sequence_train_val_split=run_config[
                "strict_sequence_train_val_split"
            ],
            include_partial_sequences_train=run_config[
                "include_partial_sequences_train"
            ],
            include_partial_sequences_val=run_config["include_partial_sequences_val"],
            random_state=random_state,
        )

        if unsupervised:
            model = train_model_unsupervised(config=run_config)
        else:
            model = train_model(config=run_config)

        mean_clustering_error = evaluate_model_performance(
            model,
            main_val_loader,
            run_config["train_data"],
            generate_video=False,
        )

        run_result = {
            "run_idx": run_idx,
            "random_state": random_state,
            "mean_clustering_error": mean_clustering_error,
        }
        all_results.append(run_result)

        print(
            f"Run {run_idx + 1} completed. Mean clustering error: {mean_clustering_error:.4f}"
        )

    mean_errors = [result["mean_clustering_error"] for result in all_results]

    aggregated_metrics = {
        "num_runs": num_runs,
        "unsupervised": unsupervised,
        "mean_clustering_error_avg": np.mean(mean_errors),
        "mean_clustering_error_var": np.var(mean_errors),
        "mean_clustering_error_std": np.std(mean_errors),
        "mean_clustering_error_min": np.min(mean_errors),
        "mean_clustering_error_max": np.max(mean_errors),
        "all_results": all_results,
    }

    print("\n=== Aggregated Metrics ===")
    print(f"Number of runs: {num_runs}")
    print(
        f"Average mean clustering error: {aggregated_metrics['mean_clustering_error_avg']:.4f}"
    )
    print(
        f"Variance of mean clustering error: {aggregated_metrics['mean_clustering_error_var']:.6f}"
    )
    print(
        f"Standard deviation of mean clustering error: {aggregated_metrics['mean_clustering_error_std']:.4f}"
    )
    print(
        f"Minimum mean clustering error: {aggregated_metrics['mean_clustering_error_min']:.4f}"
    )
    print(
        f"Maximum mean clustering error: {aggregated_metrics['mean_clustering_error_max']:.4f}"
    )

    # save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        results_file = os.path.join(
            results_dir, f"multiple_runs_results_{timestamp}{name}.json"
        )

        json_ready_metrics = copy.deepcopy(aggregated_metrics)
        json_ready_metrics["mean_clustering_error_avg"] = float(
            json_ready_metrics["mean_clustering_error_avg"]
        )
        json_ready_metrics["mean_clustering_error_var"] = float(
            json_ready_metrics["mean_clustering_error_var"]
        )
        json_ready_metrics["mean_clustering_error_std"] = float(
            json_ready_metrics["mean_clustering_error_std"]
        )
        json_ready_metrics["mean_clustering_error_min"] = float(
            json_ready_metrics["mean_clustering_error_min"]
        )
        json_ready_metrics["mean_clustering_error_max"] = float(
            json_ready_metrics["mean_clustering_error_max"]
        )

        for result in json_ready_metrics["all_results"]:
            result["mean_clustering_error"] = float(result["mean_clustering_error"])

        with open(results_file, "w") as f:
            json.dump(json_ready_metrics, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    return aggregated_metrics


def load_and_analyze_results(results_file: str) -> Dict[str, Any]:
    """
    Load and analyze results from a previous multiple runs experiment.

    Args:
        results_file: Path to the JSON file containing results

    Returns:
        Dictionary containing the loaded results
    """
    with open(results_file, "r") as f:
        results = json.load(f)

    print("\n=== Aggregated Metrics from Saved Results ===")
    print(f"Number of runs: {results['num_runs']}")
    print(f"Average mean clustering error: {results['mean_clustering_error_avg']:.4f}")
    print(
        f"Variance of mean clustering error: {results['mean_clustering_error_var']:.6f}"
    )
    print(
        f"Standard deviation of mean clustering error: {results['mean_clustering_error_std']:.4f}"
    )
    print(f"Minimum mean clustering error: {results['mean_clustering_error_min']:.4f}")
    print(f"Maximum mean clustering error: {results['mean_clustering_error_max']:.4f}")

    return results
