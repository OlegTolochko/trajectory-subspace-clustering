def load_sweep_config(
    method="bayes",
    optimize_loss_weights=True,
    use_individual_shift_augmentation=True,
    use_full_shift_augmentation=True,
    occlude_points=True,
    occlude_chunkwise=True,
    optimize_lr=True,
    optimize_dropout_rate=True,
):
    sweep_config = {"method": method}

    eval_metric = {"name": "mean_clustering_error", "goal": "minimize"}
    parameters_dict = {}

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

    if occlude_chunkwise:
        parameters_dict.update(
            {
                "augmentation_chunkwise_occlusion_percent": {
                    "values": [0.0, 0.1, 0.2, 0.3]
                },
                "augmentation_chunkwise_occlusion_max_chunk_amount": {
                    "values": [1, 2, 3, 4]
                },
            }
        )

    if optimize_lr:
        parameters_dict.update(
            {
                "learning_rate": {"values": [0.00025, 0.0001]},
                "weight_decay": {"values": [2.5e-5, 1e-5, 5e-6, 1e-6]},
                "scheduler_gamma": {"values": [0.9995, 0.999]},
            }
        )

    if optimize_dropout_rate:
        parameters_dict.update(
            {
                "dropout_rate": {"values": [0.0, 0.05, 0.1]},
            }
        )

    sweep_config["metric"] = eval_metric
    sweep_config["parameters"] = parameters_dict

    return sweep_config
