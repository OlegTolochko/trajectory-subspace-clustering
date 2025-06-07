import wandb
import typer

from training import train_model
from sweep import load_sweep_config

app = typer.Typer()

# base setup according to paper
TRAIN_CONFIG = {
    "model_name": "model",
    "pretraining_epochs": 100,
    "full_epochs": 200,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "scheduler_gamma": 0.999,
    "batch_size": 1,
    "train_data": "Hopkins155",  # Options: Hopkins155, Hopkins12, KT3DMoSeg
    "device": "cuda",  # Options: cuda, cpu, mps
    "alph0": False,  # zero-out first part of basis-function term
    "include_ortho_loss": False,  # include loss, that enforces orthogonality for basis vectors
    "include_feat_loss": True,  # include loss, that compares original trajectories w. reconstructed
    "use_sequence_randomization": False,  # randomize sequences for a class to enforce reconstruction of an. sequence
    "w_info": 1.0,  # weight for InfoNCE loss
    "w_res": 1.0,  # weight for residual loss
    "w_feat": 1.0,  # weight for feature difference loss
    "w_ortho": 0.01,  # weight for orthogonality loss
}


@app.command()
def train(sweep: bool = False):
    config = TRAIN_CONFIG
    with wandb.init(project="trajectory-subspace-clustering", config=config):
        if sweep:
            for key in wandb.config.as_dict():
                config[key] = wandb.config.as_dict().get(key)

        model = train_model(config=config)


@app.command()
def sweep(sweep_id: str = "", count: int = 20):
    sweep_config = load_sweep_config()
    if sweep_id == "":
        sweep_id = wandb.sweep(sweep_config, project="trajectory-subspace-clustering")

    wandb.agent(
        sweep_id,
        function=lambda: train(sweep=True),
        project="trajectory-subspace-clustering",
        count=count,
    )


if __name__ == "__main__":
    app()
