# base setup according to paper
CONFIG = {
    "model_name": "model",
    "pretraining_epochs": 50,
    "full_epochs": 50,
    "learning_rate": 0.00025,
    "weight_decay": 1e-5,
    "scheduler_gamma": 0.999,
    "batch_size": 1,
    "dropout_rate": 0.1,
    "validation_split": 0.2,  # if set to 1.0, the val set will be the full training set
    "random_state": 42,
    "validation_frequency": 2,  # set to 1 if validation should be performed every single time
    "include_partial_sequences_train": True,  # Dataset also includes partial versions of sequences, True to include
    "include_partial_sequences_val": True,
    "strict_sequence_train_val_split": True,  # enforces, that associated partial sequences are not mixed between train and val data
    "train_data": "Hopkins155",  # Options: Hopkins155, Hopkins12, KT3DMoSeg
    "additional_val_data": None,  # Options: None, Hopkins155, Hopkins12, KT3DMoSeg
    "generate_video_from_last_val_run": False,  # Generates video of point clusters on top of original video in last val run
    "device": "cuda",  # Options: cuda, cpu, mps
    "alph0": False,  # zero-out first part of basis-function term
    "use_sequence_randomization": False,  # randomize sequences for a class to enforce reconstruction of an. sequence
    "w_info": 0.1,  # weight for InfoNCE loss
    "w_res": 1.0,  # weight for residual loss
    "w_feat": 1.0,  # weight for feature difference loss
    "w_ortho": 0.0,  # weight for orthogonality loss
    "augmentation_individual_max_shift_amount": 0.0,  # Maximum training individual data point shift amount, range 0-1, 0 = no augmetation
    "augmentation_individual_shift_percent": 0.0,  # Percentage of training data points to shift, range 0-1, 0 = no augmetation
    "augmentation_trajectory_shift_percent": 0.0,
    "augmentation_trajectory_max_shift_amount": 0.0,
    "augmentation_occlusion_percent": 0.0,  # Percentage of training data points to occlude, range 0-1, 0 = no augmetation
    "augmentation_full_max_shift_amount": 0,  # Maximum shift amount for shifting whole sequence, range 0-1, 0 = no augmetation
    "augmentation_chunkwise_occlusion_percent": 0.0,  # Percentage of points to occlude chunkwise
    "augmentation_chunkwise_occlusion_max_chunk_amount": 0,  # Maximum amount of chunks for chunkwise occlusion
}