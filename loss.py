import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms as iT
from torchaudio import transforms as aT
from utils import generate_binary_combinations, swap_halves, dual_batch_indice_permutation

def cosine_sim(y, temperature=0.1):
    """ Scoring function (cosine similarity)."""
    y_norm = F.normalize(y, dim=-1)
    similarity_matrix = torch.matmul(y_norm, y_norm.T) / temperature
    return similarity_matrix

def image_aug(x):
    """
    Returns augmentation function for tensors of images.
    """
    augmentations = iT.Compose([
        iT.RandomRotation(10),
        iT.RandomHorizontalFlip(),
        iT.RandomResizedCrop(28, scale=(0.8, 1.0), antialias=True),
        # transforms.Lambda(lambda x: x.view(x.size(0), -1)),  # Flatten the image while keeping the first dimension
    ])
    # if len(x.shape) == 4: 
    #     aug_x = torch.stack([augmentations(x_i) for x_i in x]) # Would like to apply fresh augmentations to each image
    # else:
    #     aug_x = augmentations(x)
    aug_x = augmentations(x)
    return aug_x
# TODO: Speed up above, think it's not optimal

def audio_aug(x):
    """
    Returns augmentation function for audio.
    """
    # augmentations = iT.Lambda(lambda x: x)
    augmentations = nn.Sequential(
        aT.TimeMasking(time_mask_param=10, p=0.2, iid_masks=True),  # apply time masking
        aT.FrequencyMasking(freq_mask_param=4, iid_masks=True),  # apply frequency masking
    )
    return augmentations(x)

def aug(batch, device=None):
    images, audio = batch
    # Create two augmented versions of each image 
    image1 = image_aug(images)
    image2 = image_aug(images)
    # Create two augmented versions of each audio
    audio1 = audio_aug(audio)
    audio2 = audio_aug(audio)
    if device is not None:
        image1 = image1.to(device)
        image2 = image2.to(device)
        audio1 = audio1.to(device)
        audio2 = audio2.to(device)
    return (image1, audio1), (image2, audio2)
# TODO: pass augmentaion selection, rename above image_audio_aug

def info_critic(model, batch1, batch2, temperature, device, acc=False):
    num_modalities = len(batch1) if not torch.is_tensor(batch1) else 1
    to_disturb = generate_binary_combinations(num_modalities)
    # Create u, the anchors
    u = model(batch1)
    vs = []
    if num_modalities == 1:
        vs.append(model(batch2))
        v = batch2.roll(1, 0)
        v = model(v)
        vs.append(v)
    else:
        for disturb in to_disturb:
            v = [batch2[i].roll(1, 0) if disturb[i] else batch2[i] for i in range(num_modalities)]
            v = model(v)
            vs.append(v)

    scores = [model.score(u, v) for v in vs]

    if acc:
        # Compute accuracy
        predictions = torch.cat(scores, dim=0).argmax(dim=1)
        labels = torch.cat([i*torch.ones(s.shape[0]) for i, s in enumerate(scores)], dim=0).to(device)
        accuracy = (predictions == labels).float().mean()
        #REMOVE
        if len(scores) == 4:
            acc_0 = (scores[0].argmax(dim=1).to(device) == torch.zeros(scores[0].shape[0]).to(device)).float().mean()
            acc_1 = ( (scores[1].argmax(dim=1).to(device) == torch.ones(scores[1].shape[0]).to(device)) | (scores[1].argmax(dim=1).to(device) == 2*torch.ones(scores[1].shape[0]).to(device)) ).float().mean()
            acc_2 = ( (scores[2].argmax(dim=1).to(device) == torch.ones(scores[2].shape[0]).to(device)) | (scores[2].argmax(dim=1).to(device) == 2*torch.ones(scores[2].shape[0]).to(device)) ).float().mean()
            acc_3 = (scores[3].argmax(dim=1).to(device) == 3*torch.ones(scores[3].shape[0]).to(device)).float().mean()
            class_accs = (acc_0, acc_1, acc_2, acc_3)
            symmmetric_acc = (acc_0 + acc_1 + acc_2 + acc_3) / 4
        else:
            acc_0 = (scores[0].argmax(dim=1).to(device) == torch.zeros(scores[0].shape[0]).to(device)).float().mean()
            acc_1 = (scores[1].argmax(dim=1).to(device) == torch.ones(scores[1].shape[0]).to(device)).float().mean()
            class_accs = (acc_0, acc_1)
            symmmetric_acc = 0.0
        return accuracy, symmmetric_acc, class_accs

    # We then compute the cross entropy loss between the scores and the correct logits
    losses = [F.cross_entropy(s, torch.empty(s.shape[0], dtype=torch.long).fill_(i).to(device), reduction='mean') for i, s in enumerate(scores)]
    # We then sum the losses and return them
    loss = sum(losses) / len(losses)
    return loss

def info_critic_plus(model, batch1, batch2, temperature, device, acc=False):
    num_modalities = len(batch1) if not torch.is_tensor(batch1) else 1
    to_disturb = generate_binary_combinations(num_modalities)
    # Create u, the anchors
    u = torch.cat([model(batch1), model(batch2)], dim=0)
    loss = []
    scores = []
    if num_modalities == 1:
        v_pos = swap_halves(u)
        score = model.score(u, v_pos)
        if acc: 
            scores.append(score)
        else:
            loss.append(F.cross_entropy(score, torch.empty(score.shape[0], dtype=torch.long).fill_(0).to(device), reduction='mean'))


        neg_perm = dual_batch_indice_permutation(batch1.shape[0])
        score = model.score(u, u[neg_perm])
        if acc: 
            scores.append(score)
        else:
            loss.append(F.cross_entropy(score, torch.empty(score.shape[0], dtype=torch.long).fill_(1).to(device), reduction='mean'))

    else:
        label = -1
        for disturb in to_disturb:
            label += 1
            perm = dual_batch_indice_permutation(batch1[0].shape[0])
            v = []
            for i in range(num_modalities):
                if disturb[i]:
                    v.append(torch.cat([batch1[i], batch2[i]])[perm])
                else:
                    v.append(torch.cat([batch1[i], batch2[i]]))
            v = model(v)
            score = model.score(u, v)
            if acc: 
                scores.append(score)
            else:
                loss.append(F.cross_entropy(score, torch.empty(score.shape[0], dtype=torch.long).fill_(label).to(device), reduction='mean'))

    if acc:
        # Compute accuracy
        predictions = torch.cat(scores, dim=0).argmax(dim=1)
        labels = torch.cat([i*torch.ones(s.shape[0]) for i, s in enumerate(scores)], dim=0).to(device)
        accuracy = (predictions == labels).float().mean()
        return accuracy

    return sum(loss) / len(loss)

def info_rank(model, image_batch, audio_batch, temperature, device):
    image1, audio1, image2, audio2 = aug(image_batch, audio_batch)
    images = torch.cat([image1, image2], dim=0)
    audios = torch.cat([audio1, audio2], dim=0)

    n = len(image_batch)
    k = 1 # Num of negatives per anchor
    loss = []
    for i in range(2*n):
        # Get samples
        anchor = images[i%(2*n)], audios[i%(2*n)]

        positive = images[(n+i)%(2*n)], audios[(n+i)%(2*n)]

        possible_neg_idxs = [j for j in range(2*n) if j not in [i%(2*n), (n+i)%(2*n)]] 

        neg_idxs =  possible_neg_idxs[torch.randperm(len(possible_neg_idxs))[:k]]
        negatives = images[neg_idxs], audios[neg_idxs]

        zero_dist_idxs = possible_neg_idxs[torch.randperm(len(possible_neg_idxs))[:k]]
        zero_disturbed = images[zero_dist_idxs], torch.cat(k*[anchor[1]])

        one_dist_idxs = possible_neg_idxs[torch.randperm(len(possible_neg_idxs))[:k]]
        one_disturbed = torch.cat(k*[anchor[0]]), audios[one_dist_idxs]

        # Get encodings
        input_images = torch.stack([anchor[0], positive[0], negatives[0], zero_disturbed[0], one_disturbed[0]])
        input_audios = torch.stack([anchor[1], positive[1], negatives[1], zero_disturbed[1], one_disturbed[1]])
        encodings = model(input_images, input_audios)

        # Get scores
        score_pos = F.cosine_similarity(encodings[0], encodings[1:2]) /temperature
        score_neg = F.cosine_similarity(encodings[0], encodings[2:2+k]) /temperature
        score_zero_dist = F.cosine_similarity(encodings[0], encodings[2+k:2+2*k]) /temperature
        score_one_dist = F.cosine_similarity(encodings[0], encodings[2+2*k:2+3*k]) /temperature
        
        # Get loss
        S = torch.cat([score_pos, score_neg, score_zero_dist, score_one_dist], dim=0)
        loss.append(F.cross_entropy(S, torch.tensor(0, device=device)))
    return torch.stack(loss).mean()
# TODO: increase k from 1 and repeat positives 
# TODO: Should we be using target i for all 1,...,C? Or is just 0 every time correct?
# TODO: change from cosine similarity to learnable

def info_rank_plus(model, image_batch, audio_batch, temperature, device):
    image1, audio1, image2, audio2 = aug(image_batch, audio_batch)
    image1, audio1 = model.encode_modalities(image1, audio1)
    image2, audio2 = model.encode_modalities(image2, audio2)
    images = torch.cat([image1, image2], dim=0)
    audios = torch.cat([audio1, audio2], dim=0)

    n = len(image_batch)
    k = 100 # Num of negatives per anchor
    loss = []
    for i in range(2*n):
        # Get samples
        anchor = images[i%(2*n)], audios[i%(2*n)]

        positive = images[(n+i)%(2*n)], audios[(n+i)%(2*n)]

        possible_neg_idxs = torch.tensor([j for j in range(2*n) if j not in [i%(2*n), (n+i)%(2*n)]]) 
        neg_idxs =  torch.index_select(possible_neg_idxs, 0, torch.randperm(len(possible_neg_idxs))[:k])
        negatives = images[neg_idxs], audios[neg_idxs]

        zero_dist_idxs = torch.index_select(possible_neg_idxs, 0, torch.randperm(len(possible_neg_idxs))[:k])
        zero_disturbed = images[zero_dist_idxs], torch.cat(k*[anchor[1].unsqueeze(0)])

        one_dist_idxs = torch.index_select(possible_neg_idxs, 0, torch.randperm(len(possible_neg_idxs))[:k])
        one_disturbed = torch.cat(k*[anchor[0].unsqueeze(0)]), audios[one_dist_idxs]

        # Get encodings
        input_images = torch.cat([anchor[0].unsqueeze(0), 
                                    positive[0].unsqueeze(0), 
                                    negatives[0], 
                                    zero_disturbed[0], 
                                    one_disturbed[0]]
                                ,dim=0)
        input_audios = torch.cat([anchor[1].unsqueeze(0), 
                                    positive[1].unsqueeze(0), 
                                    negatives[1], 
                                    zero_disturbed[1], 
                                    one_disturbed[1]]
                                ,dim=0)
        encodings = model.fuse(input_images, input_audios)

        # Get scores
        score_pos = F.cosine_similarity(encodings[0], encodings[1:2]) /temperature
        score_neg = F.cosine_similarity(encodings[0], encodings[2:2+k]) /temperature
        score_zero_dist = F.cosine_similarity(encodings[0], encodings[2+k:2+2*k]) /temperature
        score_one_dist = F.cosine_similarity(encodings[0], encodings[2+2*k:2+3*k]) /temperature

        # Get loss
        S = torch.cat([score_pos, score_neg, score_zero_dist, score_one_dist], dim=0)
        loss.append(F.cross_entropy(S, torch.tensor(0, device=device)))
    return torch.stack(loss).mean()
# TODO: Should we be using target i for all 1,...,C? Or is just 0 every time correct?
# TODO: change from cosine similarity to learnable

def rnd_idx_without(start, end, idxs):
    x = torch.tensor([i for i in range(start, end) if i not in idxs])
    return x[torch.randint(0, len(x), (1,)).item()]

def prob_loss(model, image_batch, audio_batch, temperature, device):
    # Set up parameters
    n_b = image_batch.shape[0]
    n = len(image_batch)*4
    pi_0, pi_1, pi_2, pi_3 = 0.10, 0.30, 0.30, 0.30
    n_0, n_1, n_2, n_3 = int(n*pi_0), int(n*pi_1), int(n*pi_2), int(n*pi_3)

    # Get samples
    pos_idxs = torch.randperm(n_b*2)[:n_0] 
    pos_idxs = torch.stack([pos_idxs, (pos_idxs+n_b)%(2*n_b)])

    neg_idxs = torch.randperm(n_b*2)[:n_1]
    neg_idxs = torch.stack([neg_idxs, torch.tensor([rnd_idx_without(0, n_b*2, [i, (n_b+i)%(2*n_b)]) for i in neg_idxs])])

    zero_dist_idxs = torch.randperm(image_batch.shape[0]*2)[:n_2]
    zero_dist_idxs = torch.stack([zero_dist_idxs, torch.tensor([rnd_idx_without(0, n_b*2, [i, (n_b+i)%(2*n_b)]) for i in zero_dist_idxs])])

    one_dist_idxs = torch.randperm(image_batch.shape[0]*2)[:n_3]
    one_dist_idxs = torch.stack([one_dist_idxs, torch.tensor([rnd_idx_without(0, n_b*2, [i, (n_b+i)%(2*n_b)]) for i in one_dist_idxs])])

    # Find encodings
    image1, audio1, image2, audio2 = aug(image_batch, audio_batch)
    image1, audio1 = model.encode_modalities(image1, audio1)
    image2, audio2 = model.encode_modalities(image2, audio2)
    images = torch.cat([image1, image2], dim=0)
    audios = torch.cat([audio1, audio2], dim=0)

    # Find scores
    score_2 = model.score(
        model.fuse(images[zero_dist_idxs[0]], audios[zero_dist_idxs[0]]),
        model.fuse(images[zero_dist_idxs[0]], audios[zero_dist_idxs[1]]),
    )
    score_3 = model.score(
        model.fuse(images[one_dist_idxs[0]], audios[one_dist_idxs[0]]),
        model.fuse(images[one_dist_idxs[1]], audios[one_dist_idxs[0]]),
    )
    fused = model.fuse(images, audios)
    score_0 = model.score(fused[pos_idxs[0]], fused[pos_idxs[1]])
    score_1 = model.score(fused[neg_idxs[0]], fused[neg_idxs[1]])

    # Compute loss
    score_0 = pi_0 * F.cross_entropy(score_0, torch.empty(score_0.shape[0], dtype=torch.long).fill_(0).to(device))
    score_1 = pi_1 * F.cross_entropy(score_1, torch.empty(score_1.shape[0], dtype=torch.long).fill_(1).to(device))
    score_2 = pi_2 * F.cross_entropy(score_2, torch.empty(score_2.shape[0], dtype=torch.long).fill_(2).to(device))
    score_3 = pi_3 * F.cross_entropy(score_3, torch.empty(score_3.shape[0], dtype=torch.long).fill_(3).to(device))
    return score_0 + score_1 + score_2 + score_3

def decomposed_loss(model, image_batch, audio_batch, temperature, device):
    n = image_batch.shape[0]
    # Augment each sample twice (for both modalities)
    image1, audio1, image2, audio2 = aug(image_batch, audio_batch)
    image1, audio1 = model.encode_modalities(image1, audio1)
    image2, audio2 = model.encode_modalities(image2, audio2)
    
    anchor_reps = model.fuse(torch.cat([image1, image2], dim=0), torch.cat([audio1, audio2], dim=0))
    images = torch.cat([image1, image2], dim=0)
    audios = torch.cat([audio1, audio2], dim=0)

    # Positives
    pos_image_scores = model.score_image(anchor_reps, torch.cat([anchor_reps[n:2*n], anchor_reps[:n]], dim=0))
    pos_audio_scores = model.score_audio(anchor_reps, torch.cat([anchor_reps[n:2*n], anchor_reps[:n]], dim=0))
    pos_loss = F.cross_entropy(torch.cat([pos_image_scores, pos_audio_scores], dim=1), torch.repeat_interleave(torch.tensor([0, 0]), n).to(device))

    # Negatives
    neg_idxs = torch.arange(0, n*2)
    neg_idxs = torch.tensor([rnd_idx_without(0, n*2, [i, (n+i)%(2*n)]) for i in neg_idxs])
    neg_image_scores = model.score_image(anchor_reps, anchor_reps[neg_idxs])
    neg_audio_scores = model.score_audio(anchor_reps, anchor_reps[neg_idxs])
    neg_loss = F.cross_entropy(torch.cat([neg_image_scores, neg_audio_scores], dim=1), torch.repeat_interleave(torch.tensor([1, 1]), n).to(device))

    # Zero disturbed
    zero_dist_idxs = torch.arange(0, n*2)
    zero_dist_idxs = torch.tensor([rnd_idx_without(0, n*2, [i, (n+i)%(2*n)]) for i in zero_dist_idxs])
    zero_dist_image_scores = model.score_image(anchor_reps, model.fuse(images[zero_dist_idxs], audios))
    zero_loss = F.cross_entropy(torch.cat([zero_dist_image_scores, pos_audio_scores], dim=1), torch.repeat_interleave(torch.tensor([1, 0]), n).to(device))

    # One disturbed
    one_dist_idxs = torch.arange(0, n*2)
    one_dist_idxs = torch.tensor([rnd_idx_without(0, n*2, [i, (n+i)%(2*n)]) for i in one_dist_idxs])
    one_dist_audio_scores = model.score_audio(anchor_reps, model.fuse(images, audios[one_dist_idxs]))
    one_loss = F.cross_entropy(torch.cat([pos_image_scores, one_dist_audio_scores], dim=1), torch.repeat_interleave(torch.tensor([0, 1]), n).to(device))

    # Compute loss
    return (pos_loss + neg_loss + zero_loss + one_loss)/4
    # TODO: think need to work on fused 
   

def SimCLR_loss(model, batch1, batch2, temperature, device, acc=False):
    batch_size = batch1[0].shape[0] if isinstance(batch1, (tuple, list)) else batch1.shape[0]
    LARGE_NUM = 1e9
    y1, y2 = model(batch1), model(batch2)

    labels = F.one_hot(torch.arange(batch_size), num_classes=batch_size * 2).float().to(device)
    masks = F.one_hot(torch.arange(batch_size), num_classes=batch_size).float().to(device)

    logits_aa = torch.matmul(y1, y1.t()) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(y2, y2.t()) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(y1, y2.t()) / temperature
    logits_ba = torch.matmul(y2, y1.t()) / temperature

    loss_a = F.cross_entropy(
        torch.cat([logits_ab, logits_aa], dim=1),
        torch.arange(batch_size).to(device)
    )
    loss_b = F.cross_entropy(
        torch.cat([logits_ba, logits_bb], dim=1),
        torch.arange(batch_size).to(device)
    )
    loss = loss_a + loss_b

    if acc==True:
        labels = labels.argmax(dim=1)
        predictions = logits_ab.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        return accuracy
    
    return loss

def get_train_accuracy(model, batch1, batch2, est, device):
    if est == 'info_critic':
        train_acc, symmetric_acc, class_acc = info_critic(model, batch1, batch2, 1, device, acc=True)
        acc = (train_acc, symmetric_acc, class_acc) #REMOVE
    elif est == 'info_critic_plus':
        acc = info_critic_plus(model, batch1, batch2, 1, device, acc=True)
    elif est == 'SimCLR':
        acc = SimCLR_loss(model, batch1, batch2, 1, device, acc=True)
    else:
        raise Exception('Unknown estimation method')
    return acc