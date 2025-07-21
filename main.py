import os
import warnings

import wandb
import typer

from training import train_model
from training_unsupervised import train_model_unsupervised
from sweep import load_sweep_config
from inference import perform_inference

app = typer.Typer()

# base setup according to paper
CONFIG = {
    "model_name": "model",
    "pretraining_epochs": 25,
    "full_epochs": 40,
    "learning_rate": 0.00025,
    "weight_decay": 1e-5,
    "scheduler_gamma": 0.999,
    "batch_size": 1,
    "dropout_rate": 0.1,
    "validation_split": 1.0,  # if set to 1.0, the val set will be the full training set
    "include_partial_sequences_train": True,  # Dataset also includes partial versions of sequences, True to include
    "include_partial_sequences_val": True,
    "strict_sequence_train_val_split": True,  # enforces, that associated partial sequences are not mixed between train and val data
    "train_data": "Hopkins155",  # Options: Hopkins155, Hopkins12, KT3DMoSeg
    "additional_val_data": None,  # Options: None, Hopkins155, Hopkins12, KT3DMoSeg
    "generate_video_from_last_val_run": False,  # Generates video of point clusters on top of original video in last val run
    "device": "cuda",  # Options: cuda, cpu, mps
    "alph0": False,  # zero-out first part of basis-function term
    "use_sequence_randomization": False,  # randomize sequences for a class to enforce reconstruction of an. sequence
    "w_info": 1.0,  # weight for InfoNCE loss
    "w_res": 0.5,  # weight for residual loss
    "w_feat": 0.5,  # weight for feature difference loss
    "w_ortho": 0.0,  # weight for orthogonality loss
    "augmentation_individual_max_shift_amount": 0.0,  # Maximum training individual data point shift amount, range 0-1, 0 = no augmetation
    "augmentation_individual_shift_percent": 0.0,  # Percentage of training data points to shift, range 0-1, 0 = no augmetation
    "augmentation_occlusion_percent": 0.0,  # Percentage of training data points to occlude, range 0-1, 0 = no augmetation
    "augmentation_full_max_shift_amount": 0.0,  # Maximum shift amount for shifting whole sequence, range 0-1, 0 = no augmetation
    "augmentation_chunkwise_occlusion_percent": 0.0,  # Percentage of points to occlude chunkwise
    "augmentation_chunkwise_occlusion_max_chunk_amount": 0,  # Maximum amount of chunks for chunkwise occlusion
}


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


if __name__ == "__main__":
    app()
