import torch.nn as nn
import torch.nn.functional as F
import torch

def score_fun(y, temperature=0.1):
    """ Scoring function (cosine similarity)."""
    y_norm = F.normalize(y, dim=-1)
    similarity_matrix = torch.matmul(y_norm, y_norm.T) / temperature
    return similarity_matrix

def aug(x1, x2):
    # For each x1_i in x1 we return [x1_i, x1_i, x1_i+1, x1_i+1] and for each x2_i in x2 we return [x2_i, x2_i+1, x2_i, x2_i+1]
    x1_aug = torch.cat([x1, x1, x1.roll(1, 0), x1.roll(1, 0)], dim=0)
    x2_aug = torch.cat([x2, x2.roll(1, 0), x2, x2.roll(1, 0)], dim=0)
    return x1_aug, x2_aug
 

def info_rank_loss(model, image_batch, audio_batch, temperature, device):
    aug_1, aug_2 = aug(image_batch, audio_batch)
    h_x = model(aug_1, aug_2)
    reshape h_x to n*n 

    softmax/score ( diag + row[i] col[i]) multipying by pi_c

    score_matrix = score_fun(h_x, temperature)

    logits = nn.LogSoftmax(dim=1)(h_x)
    loss = -torch.sum((1/4) * logits.mean(dim=0))
    return loss