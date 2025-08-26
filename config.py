# base setup according to paper

CONFIG = {
    """Base setup (unspecified refers to not specified in original paper):"""
    "model_name": "model",
    "pretraining_epochs": 50,  # unspecified
    "full_epochs": 50,  # unspecified
    "learning_rate": 0.00025,  # original paper used 0.001
    "weight_decay": 1e-5,  # unspecified
    "scheduler_gamma": 0.999,
    "batch_size": 1,
    "validation_split": 0.2,  # unspecified; if set to 1.0, the val set will be the full training set
    "random_state": 42,  # unspecified
    "w_info": 0.1,  # unspecified; weight for InfoNCE loss
    "w_res": 1.0,  # unspecified; weight for residual loss
    "w_feat": 1.0,  # unspecified; weight for feature difference loss
    "device": "cuda",  # Options: cuda, cpu, mps
    """Additional adjustable parameters:"""
    "transformer_encoder_feature_extractor": False,  # If True utilizes different Feature Extractor with an TransformerEncoder layer
    "validation_frequency": 1,  # set to 1 if validation should be performed every single time
    "include_partial_sequences_train": True,  # Train set also includes partial versions of sequences, True to include
    "include_partial_sequences_val": True,  # Validation set also includes partial versions of sequences, True to include
    "strict_sequence_train_val_split": True,  # Enforces, that associated partial sequences are not mixed between train and val data
    "train_data": "Hopkins155",  # Options: Hopkins155, Hopkins12, KT3DMoSeg
    "additional_val_data": None,  # Options: None, Hopkins155, Hopkins12, KT3DMoSeg
    "generate_video_from_last_val_run": False,  # Generates video of point clusters on top of original video in last val run
    "w_ortho": 0.0,  # weight for orthogonality loss
    """Augmentation Framework:"""
    "augmentation_individual_max_shift_amount": 0.0,  # Maximum training individual data point shift amount, range 0-1, 0 = no augmetation
    "augmentation_individual_shift_percent": 0.0,  # Percentage of training data points to shift, range 0-1, 0 = no augmetation
    "augmentation_trajectory_individual_shift_percent": 0.0,
    "augmentation_trajectory_individual_shift_distance": 0.0,
    "augmentation_trajectory_shift_percent": 0.0,
    "augmentation_trajectory_max_shift_amount": 0.0,
    "augmentation_occlusion_percent": 0.0,  # Percentage of training data points to occlude, range 0-1, 0 = no augmetation
    "augmentation_full_max_shift_amount": 0,  # Maximum shift amount for shifting whole sequence, range 0-1, 0 = no augmetation
    "augmentation_chunkwise_occlusion_percent": 0.0,  # Percentage of points to occlude chunkwise
    "augmentation_chunkwise_occlusion_max_chunk_amount": 0,  # Maximum amount of chunks for chunkwise occlusion
}
