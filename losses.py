import torch
import torch.nn.functional as F

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
    squared_norms = torch.sum(X ** 2, dim=1, keepdim=True)  # (B, 1)
    dists = squared_norms - 2 * dot_product + squared_norms.T  # (B, B)
    return dists


def L_Residual(x_original, x_reconstructed):
    loss = F.mse_loss(x_original, x_reconstructed, reduction='mean')
    return loss

# same implementation as L_Residual
def L_FeatDiff(f_original, f_reconstructed):
    loss = F.mse_loss(f_original, f_reconstructed, reduction='mean')
    return loss
