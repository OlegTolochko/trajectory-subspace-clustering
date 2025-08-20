import os
import warnings

import wandb
import typer

from training import train_model
from training_unsupervised import train_model_unsupervised
from sweep import load_sweep_config
from inference import perform_inference
from config import CONFIG

app = typer.Typer()


def setup_directories():
    os.makedirs("out", exist_ok=True)
    os.makedirs("out/models", exist_ok=True)
    print("Created output directories: out/ and out/models/")


def validate_data_directories():
    data_dir = "data"

    if not os.path.exists(data_dir):
        warnings.warn(
            f"Data directory '{data_dir}' not found! Please create it and add your datasets.",
            UserWarning,
        )
        return

    expected_datasets = ["Hopkins155", "Hopkins12", "KT3DMoSeg"]

    for dataset in expected_datasets:
        dataset_path = os.path.join(data_dir, dataset)
        if not os.path.exists(dataset_path):
            warnings.warn(
                f"Dataset directory '{dataset_path}' not found. "
                f"If you plan to use {dataset}, please add the data to this directory.",
                UserWarning,
            )
        elif not os.listdir(dataset_path):
            warnings.warn(
                f"Dataset directory '{dataset_path}' exists but is empty. "
                f"Please add the {dataset} dataset files.",
                UserWarning,
            )
        else:
            print(f"Found {dataset} data in {dataset_path}")


@app.command()
def train(sweep: bool = False, unsupervised: bool = False):
    setup_directories()
    validate_data_directories()

    config = CONFIG
    with wandb.init(project="trajectory-subspace-clustering", config=config):
        if sweep:
            for key in wandb.config.as_dict():
                config[key] = wandb.config.as_dict().get(key)
        if unsupervised:
            model = train_model_unsupervised(config=config)
        else:
            model = train_model(config=config)


@app.command()
def sweep(sweep_id: str = "", count: int = 100):
    setup_directories()
    validate_data_directories()

    sweep_config = load_sweep_config()
    if sweep_id == "":
        sweep_id = wandb.sweep(sweep_config, project="trajectory-subspace-clustering")

    wandb.agent(
        sweep_id,
        function=lambda: train(sweep=True, unsupervised=False),
        project="trajectory-subspace-clustering",
        count=count,
    )


@app.command()
def unsupervised_sweep(sweep_id: str = "", count: int = 100):
    setup_directories()
    validate_data_directories()

    sweep_config = load_sweep_config()
    if sweep_id == "":
        sweep_id = wandb.sweep(sweep_config, project="trajectory-subspace-clustering")

    wandb.agent(
        sweep_id,
        function=lambda: train(sweep=True, unsupervised=True),
        project="trajectory-subspace-clustering",
        count=count,
    )


@app.command()
def inference(model_name: str):
    validate_data_directories()

    perform_inference(model_name=model_name, config=CONFIG)


def run_flexible_config_experiment(param_config, num_runs=25, name_prefix="flexible"):
    """
    Python function version for programmatic use.
    
    Args:
        param_config: Dict with parameter names as keys and lists of values
                     e.g., {"param1": [val1, val2], "param2": [val3, val4]}
        num_runs: Number of runs per configuration
        name_prefix: Prefix for naming experiments
    
    Returns:
        List of results for each configuration
    """
    import itertools
    from multiple_runs import run_multiple_experiments
    
    param_names = list(param_config.keys())
    param_values = list(param_config.values())
    all_combinations = list(itertools.product(*param_values))
    
    all_results = []
    
    for i, combination in enumerate(all_combinations, 1):
        new_config = CONFIG.copy()
        param_dict = {}
        
        for param_name, value in zip(param_names, combination):
            new_config[param_name] = value
            param_dict[param_name] = value
        
        param_str = "_".join([f"{name}_{value}" for name, value in param_dict.items()])
        name = f"_{name_prefix}_{param_str}"
        
        print(f"Running configuration {i}/{len(all_combinations)}: {param_dict}")
        
        results = run_multiple_experiments(
            num_runs=num_runs,
            unsupervised=False,
            config=new_config,
            name=name
        )
        
        all_results.append({
            "config_id": i,
            "parameters": param_dict,
            "results": results
        })
    
    return all_results

if __name__ == "__main__":
    app()