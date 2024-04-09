import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms as iT
from torchaudio import transforms as aT

def image_aug(x):
    """
    Returns augmentation function for images.
    """
    augmentations = iT.Compose([
        iT.RandomRotation(10),
        iT.RandomHorizontalFlip(),
        iT.RandomResizedCrop(28, scale=(0.8, 1.0), antialias=True),
        # transforms.Lambda(lambda x: x.view(x.size(0), -1)),  # Flatten the image while keeping the first dimension
    ])
    return augmentations(x)

def audio_aug(x):
    """
    Returns augmentation function for audio.
    """
    augmentations = nn.Sequential(
        aT.TimeMasking(time_mask_param=10),  # apply time masking
        aT.FrequencyMasking(freq_mask_param=5),  # apply frequency masking
        # aT.TimeStretch(n_freq=13),  # apply time stretching
        # transforms.Lambda(lambda x: x.view(x.size(0), -1)),  # Flatten the audio while keeping the first dimension
    )
    return augmentations(x)

def aug(images, audio):
    # Create two augmented versions of each image 
    image1 = image_aug(images)
    image2 = image_aug(images)
    # Create two augmented versions of each audio
    audio1 = audio_aug(audio)
    audio2 = audio_aug(audio)
    return image1, audio1, image2, audio2

def info_rank_loss(model, image_batch, audio_batch, temperature, device):
    image1, audio1, image2, audio2 = aug(image_batch, audio_batch)
    # Create u, the anchors (may augment this later)
    u = model(image1 , audio1)
    # Create v0, the positive samples p(y|x_1,x_2)
    v0 = model(image2, audio2)
    # Create v1, the negative samples p(y)
    v1 = v0.roll(1, 0)
    # Create v2, the disturbed negative samples p(y|x1)
    v2 = model(image2, audio2.roll(1, 0))
    # Create v3, the disturbed negative samples p(y|x2)
    v3 = model(image2.roll(1, 0), audio2)

    # We then compute scores between u and v0, u and v1, u and v2, u and v3
    score0 = model.score(u, v0, temperature)
    score1 = model.score(u, v1, temperature)
    score2 = model.score(u, v2, temperature)
    score3 = model.score(u, v3, temperature)

    # We then compute the cross entropy loss between the scores and the correct logits
    loss0 = F.cross_entropy(score0, torch.empty(score0.shape[0], dtype=torch.long).fill_(0).to(device), reduction='mean')
    loss1 = F.cross_entropy(score1, torch.empty(score1.shape[0], dtype=torch.long).fill_(1).to(device), reduction='mean')
    loss2 = F.cross_entropy(score2, torch.empty(score2.shape[0], dtype=torch.long).fill_(2).to(device), reduction='mean')
    loss3 = F.cross_entropy(score3, torch.empty(score3.shape[0], dtype=torch.long).fill_(3).to(device), reduction='mean')
    # We then sum the losses and return them
    loss = (loss0 + loss1 + loss2 + loss3) / 4
    print(loss)
    return loss

# TODO:
    # Should we average the losses?
    # Do we need to weigh with pi_c's in the cross entropy loss?
    # Softmax?
    # Exponential? / Logarithm?