def load_sweep_config(
    method="bayes",
    optimize_loss_weights=True,
    use_individual_shift_augmentation=True,
    use_full_shift_augmentation=True,
    occlude_points=True,
):
    sweep_config = {"method": method}

    eval_metric = {"name": "mean_clustering_error", "goal": "minimize"}
    parameters_dict = {
        "validation_split": {"values": [0.2]},
        "pretraining_epochs": {"values": [30]},
        "full_epochs": {"values": [60]},
        "learning_rate": {"values": [0.001]},
        "weight_decay": {"values": [1e-5]},
        "scheduler_gamma": {"values": [0.999]},
    }

    if optimize_loss_weights:
        parameters_dict.update(
            {
                "w_info": {"values": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]},
                "w_res": {"values": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]},
                "w_feat": {"values": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]},
                "w_ortho": {"values": [0.0, 0.001, 0.005, 0.01, 0.025, 0.05]},
            }
        )

    if use_individual_shift_augmentation:
        parameters_dict.update(
            {
                "augmentation_individual_max_shift_amount": {"values": [0.1, 0.2, 0.3]},
                "augmentation_individual_shift_percent": {
                    "values": [0.0, 0.1, 0.2, 0.3]
                },
            }
        )

    if use_full_shift_augmentation:
        parameters_dict.update(
            {
                "augmentation_full_max_shift_amount": {"values": [0, 0.1, 0.2, 0.3]},
            }
        )
    if occlude_points:
        parameters_dict.update(
            {"augmentation_occlusion_percent": {"values": [0.0, 0.1, 0.2, 0.3]}}
        )

    sweep_config["metric"] = eval_metric
    sweep_config["parameters"] = parameters_dict

    return sweep_config


def load_unsupervised_sweep_config(
    method="bayes",
    use_individual_shift_augmentation=True,
    use_full_shift_augmentation=True,
    occlude_points=True,
):
    sweep_config = {"method": method}

    eval_metric = {"name": "mean_clustering_error", "goal": "minimize"}
    parameters_dict = {
        "validation_split": {"values": [0.2]},
        "pretraining_epochs": {"values": [30]},
        "full_epochs": {"values": [60]},
        "learning_rate": {"values": [0.001]},
        "weight_decay": {"values": [1e-5]},
        "scheduler_gamma": {"values": [0.999]},
    }

    if use_individual_shift_augmentation:
        parameters_dict.update(
            {
                "augmentation_individual_max_shift_amount": {"values": [0.1, 0.2, 0.3]},
                "augmentation_individual_shift_percent": {
                    "values": [0.0, 0.1, 0.2, 0.3]
                },
            }
        )

    if use_full_shift_augmentation:
        parameters_dict.update(
            {
                "augmentation_full_max_shift_amount": {"values": [0, 0.1, 0.2, 0.3]},
            }
        )

    if occlude_points:
        parameters_dict.update(
            {"augmentation_occlusion_percent": {"values": [0.0, 0.1, 0.2, 0.3]}}
        )

    sweep_config["metric"] = eval_metric
    sweep_config["parameters"] = parameters_dict

    return sweep_config
