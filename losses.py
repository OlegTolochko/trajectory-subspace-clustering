import torch
import torch.nn.functional as F
import numpy as np


def L_InfoNCE(features, labels, temperature: float = 1.0):
    B = features.size(0)

    dists = pairwise_squared_l2(features)
    logits = -dists / temperature
    logits.fill_diagonal_(-1e9)

    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T)
    pos_mask.fill_diagonal_(False)
    loss = -log_prob[pos_mask].mean()

    return loss


def pairwise_squared_l2(X):
    dot_product = X @ X.T  # (B, B)
    squared_norms = torch.sum(X**2, dim=1, keepdim=True)  # (B, 1)
    dists = squared_norms - 2 * dot_product + squared_norms.T  # (B, B)
    return dists


def L_Residual(x_original, x_reconstructed):
    loss = F.mse_loss(x_original, x_reconstructed, reduction="mean")
    return loss


# same implementation as L_Residual
def L_FeatDiff(f_original, f_reconstructed):
    loss = F.mse_loss(f_original, f_reconstructed, reduction="mean")
    return loss


def L_orthogonal(h_t):
    batch_size, seq_len, n_basis = h_t.shape

    h_t_norm = h_t / (torch.linalg.norm(h_t, dim=1, keepdim=True) + 1e-8)
    cosine_sim_matrix = torch.bmm(h_t_norm.transpose(1, 2), h_t_norm)
    id_matrix = (
        torch.eye(n_basis, device=h_t.device)
        .unsqueeze(0)
        .expand(batch_size, n_basis, n_basis)
    )

    off_diagonal_cosine_sim = cosine_sim_matrix - id_matrix
    loss_ortho = torch.sum(off_diagonal_cosine_sim**2, dim=(1, 2)).mean()
    return loss_ortho
