import torch.nn as nn
import torch
from .feature_extractor import FeatureExtractor, FeatureExtractorTransformer
from .subspace_estimator import SubspaceEstimator


class TrajectoryEmbeddingModel(nn.Module):
    def __init__(self, include_transformer_encdoder=False):
        super(TrajectoryEmbeddingModel, self).__init__()
        if include_transformer_encdoder:
            self.feature_extractor = FeatureExtractorTransformer()
        else:
            self.feature_extractor = FeatureExtractor()
        self.subspace_estimator = SubspaceEstimator()

    def forward(self, x, t):
        # x: (Batch, SeqLen, 2)
        # t: (Batch, SeqLen)

        x_permuted = x.permute(0, 2, 1)  # (Batch, 2, SeqLen) for Conv1d
        f = self.feature_extractor(x_permuted)

        B, h_t = self.subspace_estimator(f, t)

        return f, B, h_t
