import torch
import torch.nn.functional as F

def L_InfoNCE(features, labels, temperature=1.0, eps=1e-9):
    batch_size = features.shape[0]
    dists = pairwise_squared_l2(features)
    p_matrix = torch.exp(-dists / temperature)

    labels = labels.view(-1, 1)
    mask_positive = torch.eq(labels, labels.T).float()
    mask_positive.fill_diagonal_(0.0)

    mask_denom = torch.ones_like(mask_positive)
    mask_denom.fill_diagonal_(0.0)

    log_prob_list = []
    for i in range(batch_size):
        pos_similarities = p_matrix[i][mask_positive[i].bool()]
        denom_similarities = p_matrix[i][mask_denom[i].bool()] 
        denom_sum = torch.sum(denom_similarities) + eps
        
        if pos_similarities.nelement() == 0:
            continue
        
        log_prob_p = torch.log(pos_similarities / denom_sum)
        
        num_positives = pos_similarities.nelement()
        avg_log_prob_i = log_prob_p.sum() / num_positives
        log_prob_list.append(avg_log_prob_i)

    loss = -torch.stack(log_prob_list).mean()
        
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
