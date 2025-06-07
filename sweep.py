def load_sweep_config(
    method="random", optimize_loss_weights=True, optimize_loss_inclusion=False
):
    sweep_config = {"method": method}

    eval_metric = {"name": "mean_clustering_error", "goal": "minimize"}
    parameters_dict = {
        "validation_split": {"values": [0.2]},
        "pretraining_epochs": {"values": [50]},
        "full_epochs": {"values": [100]},
        "learning_rate": {"values": [0.001]},
        "weight_decay": {"values": [1e-5]},
        "scheduler_gamma": {"values": [0.999]},
    }

    if optimize_loss_weights:
        parameters_dict.update(
            {
                "w_info": {"min": 0.1, "max": 1.0},
                "w_res": {"min": 0.1, "max": 1.0},
                "w_feat": {"min": 0.1, "max": 1.0},
                "w_ortho": {
                    "min": 0.005,
                    "max": 0.05,
                },
            }
        )

    if optimize_loss_inclusion:
        parameters_dict.update(
            {
                "include_ortho_loss": {"values": [True, False]},
                "include_feat_loss": {"values": [True, False]},
            }
        )

    sweep_config["metric"] = eval_metric
    sweep_config["parameters"] = parameters_dict

    return sweep_config
