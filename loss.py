import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import v2 as iT
from torchaudio import transforms as aT
from utils import generate_binary_combinations, swap_halves, dual_batch_indice_permutation, create_distribution, rnd_idx_without
import math
import numpy as np
import copy

def cosine_sim(y, temperature=0.1):
    """ Scoring function (cosine similarity)."""
    y_norm = F.normalize(y, dim=-1)
    similarity_matrix = torch.matmul(y_norm, y_norm.T) / temperature
    return similarity_matrix

class RandomSequentialCropResize(torch.nn.Module):
    def __init__(self, crop_ratio=0.8):
        super().__init__()
        self.crop_ratio = crop_ratio
    
    def forward(self, x):
        original_length = x.shape[1]
        crop_size = int(original_length * self.crop_ratio)
        start = torch.randint(0, original_length - crop_size + 1, (1,))
        cropped = x[:, start:start+crop_size]
        resized = torch.nn.functional.interpolate(cropped, size=original_length, mode='linear', align_corners=False).squeeze(1)
        return resized

class TimeWarp(torch.nn.Module):
    def __init__(self, warp_factor=0.2):
        super().__init__()
        self.warp_factor = warp_factor
    
    def forward(self, x):
        orig_steps = torch.arange(x.shape[1])
        warp_steps = orig_steps + torch.randn(x.shape[1]) * self.warp_factor
        warp_steps = torch.clamp(warp_steps, 0, x.shape[1] - 1)
        return torch.nn.functional.interpolate(x, size=x.shape[1], mode='linear', align_corners=False).squeeze(1)

class Jitter(torch.nn.Module):
    def __init__(self, sigma=0.03):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        return x + torch.randn_like(x) * self.sigma
    
def image_aug(x, type='mnist'):
    """
    Returns augmentation function for tensors of images.
    """
    if type == 'mnist':
        augmentations = iT.Compose([
            iT.RandomRotation(10),
            iT.RandomHorizontalFlip(),
            iT.RandomResizedCrop(28, scale=(0.8, 1.0), antialias=True),
        ])
    elif type == 'nyu-rgb':
        augmentations = iT.Compose([
            iT.RandomRotation(50),
            iT.RandomHorizontalFlip(p=0.5),
            iT.RandomVerticalFlip(p=0.5),
            iT.RandomResizedCrop((240, 320), scale=(0.8, 1.0), antialias=True),
            iT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    elif type == 'nyu-depth':
        augmentations = iT.Compose([
            iT.RandomRotation(50),
            iT.RandomHorizontalFlip(p=0.5),
            iT.RandomVerticalFlip(p=0.5),
            iT.RandomResizedCrop((240, 320), scale=(0.8, 1.0), antialias=True),
        ])
    else:
        raise ValueError("Invalid type, must be one of: 'mnist', 'nyu-rgb', 'nyu-depth'")
    # print(x.shape)
    aug_x = augmentations(x)
    return aug_x

# def feat_aug(x):
#     augmentations = iT.Compose([
#         aT.TimeMasking(time_mask_param=10, p=0.15, iid_masks=False),  # apply time masking
#         aT.FrequencyMasking(freq_mask_param=4, iid_masks=False),  # apply frequency masking
#         iT.GaussianNoise(mean=0.0, sigma=0.1, clip=False),
#     ])
#     return augmentations(x)

def permute(x):
    # shuffle the sequence order
    idx = torch.randperm(x.shape[0])
    return x[idx]

def noise(x):
    noise = torch.randn(x.shape) * 0.1
    return x + noise.to(x.device)

def drop(x):
    # drop 20% of the sequences
    drop_num = x.shape[0] // 5

    x_aug = torch.clone(x)
    drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
    x_aug[drop_idxs] = 0.0
    return x_aug 

def identity(x):
    return x

def feat_aug(x_batch):
    v1 = x_batch
    v2 = torch.clone(v1)
    transforms = [permute, noise, drop, identity]

    for i in range(x_batch.shape[0]):
        t_idxs = np.random.choice(4, 1, replace=False)
        t = transforms[t_idxs[0]]
        v2[i] = t(v2[i])

    return v2

def sentence_aug(x):
    """
    Randomly mask a random number of words in the sentence.
    x is a list of sentences, each sentence is a list of words. 
    """
    # Randomly mask a random number of words in the sentence
    x_new = copy.deepcopy(x)
    for i, sentence in enumerate(x_new):
        n = len(sentence)
        mask = np.random.choice(n, np.random.randint(n)//4, replace=False)
        for j in mask:
            sentence[j] = '[MASK]'
    return x_new
    

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

# TODO: Add feature_aug function, which is aug function but with random scaling, time warping, jittering (random noise)

def image_audio_aug(batch, device=None):
    images, audio = batch
    # Create two augmented versions of each image 
    image1 = image_aug(images, type='mnist')
    image2 = image_aug(images, type='mnist')
    # Create two augmented versions of each audio
    audio1 = audio_aug(audio)
    audio2 = audio_aug(audio)
    if device is not None:
        image1 = image1.to(device)
        image2 = image2.to(device)
        audio1 = audio1.to(device)
        audio2 = audio2.to(device)
    return (image1, audio1), (image2, audio2)

def augmenter(batch, modality, device):
    batch = [b.to(device) if torch.is_tensor(b) else b for b in batch] if not torch.is_tensor(batch) else batch.to(device) 
    if modality == 'image+audio':
        batch1, batch2 = image_audio_aug(batch[:2], device=device)
    elif modality == 'image':
        batch1, batch2 = image_aug(batch[0], type='mnist').to(device), image_aug(batch[0], type='mnist').to(device)
    elif modality == 'audio':
        batch1, batch2 = audio_aug(batch[1]).to(device), audio_aug(batch[1]).to(device)
    elif modality == 'image+depth':
        batch1, batch2 = [image_aug(batch[0], type='nyu-rgb').to(device), image_aug(batch[1], type='nyu-depth').to(device)], [image_aug(batch[0], type='nyu-rgb').to(device), image_aug(batch[1], type='nyu-depth').to(device)],
    elif modality in ['image_ft+audio_ft+text', 'image_ft+text']:
        batch1 = [feat_aug(batch[i]).to(device) for i in range(3)]
        batch2 = [feat_aug(batch[i]).to(device) for i in range(3)]
    elif modality == 'image_ft+audio_ft+text_bert':
        # sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask = batch
        visual, acoustic, sentences, labels = batch 
        batch1 = [feat_aug(visual).to(device), feat_aug(acoustic).to(device), sentence_aug(sentences)] #, bert_sentences.to(device), bert_sentence_types.to(device), bert_sentence_att_mask.to(device), lengths.to('cpu')]
        batch2 = [feat_aug(visual).to(device), feat_aug(acoustic).to(device), sentence_aug(sentences)] #, bert_sentences.to(device), bert_sentence_types.to(device), bert_sentence_att_mask.to(device), lengths.to('cpu')]
    else:
        raise ValueError("Invalid aug")
    # batch = [b.to('cpu') for b in batch] if not torch.is_tensor(batch) else batch.to('cpu') 
    return batch1, batch2

def _info_critic_acc(scores, device):
    # Compute accuracy
    predictions = torch.cat(scores, dim=0).argmax(dim=1)
    labels = torch.cat([i*torch.ones(s.shape[0]) for i, s in enumerate(scores)], dim=0).to(device)
    accuracy = (predictions == labels).float().mean()
    #REMOVE
    if len(scores) == 4:
        acc_0 = (scores[0].argmax(dim=1).to(device) == torch.zeros(scores[0].shape[0]).to(device)).float().mean()
        acc_1_sym = ( (scores[1].argmax(dim=1).to(device) == torch.ones(scores[1].shape[0]).to(device)) | (scores[1].argmax(dim=1).to(device) == 2*torch.ones(scores[1].shape[0]).to(device)) ).float().mean()
        acc_1 = (scores[1].argmax(dim=1).to(device) == torch.ones(scores[1].shape[0]).to(device)).float().mean()
        acc_2_sym = ( (scores[2].argmax(dim=1).to(device) == torch.ones(scores[2].shape[0]).to(device)) | (scores[2].argmax(dim=1).to(device) == 2*torch.ones(scores[2].shape[0]).to(device)) ).float().mean()
        acc_2 = (scores[2].argmax(dim=1).to(device) == torch.ones(scores[2].shape[0]).to(device)).float().mean()
        acc_3 = (scores[3].argmax(dim=1).to(device) == 3*torch.ones(scores[3].shape[0]).to(device)).float().mean()
        symmmetric_acc = (acc_0 + acc_1 + acc_2 + acc_3) / 4
        accs = {'accuracy': accuracy, 
                'symmetric_accuracy': symmmetric_acc, 
                'class_0': acc_0,
                'class_1': acc_1,
                'class_2': acc_2,
                'class_3': acc_3,
                'class_1_symmetric': acc_1_sym,
                'class_2_symmetric': acc_2_sym,
                }
    else:
        acc_0 = (scores[0].argmax(dim=1).to(device) == torch.zeros(scores[0].shape[0]).to(device)).float().mean()
        acc_1 = (scores[1].argmax(dim=1).to(device) == torch.ones(scores[1].shape[0]).to(device)).float().mean()
        accs = {'accuracy': accuracy,
                'class_0': acc_0,
                'class_1': acc_1
                }
    return accs

# def info_critic(model, batch1, batch2, temperature, device, acc=False):
#     num_modalities = len(batch1) if not torch.is_tensor(batch1) else 1
#     to_disturb = generate_binary_combinations(num_modalities)
#     # Create u, the anchors
#     u = model(batch1)
#     vs = []
#     if num_modalities == 1:
#         vs.append(model(batch2))
#         v = batch2.roll(1, 0)
#         v = model(v)
#         vs.append(v)
#     else:
#         for disturb in to_disturb:
#             v = [batch2[i].roll(1, 0) if disturb[i] else batch2[i] for i in range(num_modalities)]
#             v = model(v)
#             vs.append(v)
#     scores = [model.score(u, v) for v in vs]

#     if acc:
#         return _info_critic_acc(scores, device)
#     # We then compute the cross entropy loss between the scores and the correct logits
#     losses = [F.cross_entropy(s, torch.empty(s.shape[0], dtype=torch.long).fill_(i).to(device), reduction='mean') for i, s in enumerate(scores)]
#     print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024), flush=True) # TODO: Remove
#     print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024), flush=True) # TODO: Remove
#     print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024), flush=True) # TODO: Remove
#     # We then sum the losses and return them
#     loss = sum(losses) / len(losses)
#     return loss

def _info_critic_single_modality(model, batch1, batch2, device, acc=False, **kwargs):
    scores = []
    u = model(batch1)
    v = model(batch2)
    score_0 = model.score(u, v)
    scores.append(score_0)

    v = batch2.roll(1, 0)
    v = model(v)
    score_1 = model.score(u, v)
    scores.append(score_1)
    if acc:
        return _info_critic_acc(scores, device)
    loss = F.cross_entropy(score_0, torch.empty(score_0.shape[0], dtype=torch.long).fill_(0).to(device), reduction='mean')
    loss += F.cross_entropy(score_1, torch.empty(score_1.shape[0], dtype=torch.long).fill_(1).to(device), reduction='mean')
    return loss

def info_critic(model, batch1, batch2, device, acc=False, **kwargs):
    try:
        encode_batch = model.encode_batch
    except AttributeError:
        encode_batch = False

    num_modalities = len(batch1) if not torch.is_tensor(batch1) else 1
    if num_modalities == 1:
        return _info_critic_single_modality(model, batch1, batch2, device, acc)
    to_disturb = generate_binary_combinations(num_modalities)
    # Create u, the anchors
    u = model(batch1)
    if encode_batch:
        batch2 = model.encode_modalities(batch2)
    if acc:
        scores = []
    loss = 0
    for i, disturb in enumerate(to_disturb):
        v = [batch2[j].roll(1, 0) if disturb[j] else batch2[j] for j in range(num_modalities)]
        if encode_batch:
            v = model.fuse(v)
        else:
            v = model(v)
        score = model.score(u, v) 
        if acc:
            scores.append(score)
        else:
            loss = F.cross_entropy(score, torch.empty(score.shape[0], dtype=torch.long).fill_(i).to(device), reduction='mean')
            loss = loss / len(to_disturb)
            if i == len(to_disturb) - 1:
                loss = loss
            else:
                loss.backward(retain_graph=True)
    if acc:
        return _info_critic_acc(scores, device)
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
        accs = {'accuracy': accuracy}
        return accs

    return sum(loss) / len(loss)

def _prob_loss_single_modality(model, batch1, batch2, temperature, device, acc=False):
    # Set up parameters
    num_modalities = 1
    n_b = batch1.shape[0]
    n = n_b*2
    pi = create_distribution(2)
    ns = [math.ceil(n*p) for p in pi]

    pos_idxs = torch.cat([torch.stack([torch.arange(0, n_b), torch.arange(n_b, 2*n_b)], dim=0), torch.stack([torch.arange(n_b, 2*n_b), torch.arange(0, n_b)], dim=0)], dim=1)
    batch = torch.cat([batch1, batch2], dim=0)
    loss =0

    base_idxs = pos_idxs[:, torch.randperm(pos_idxs.shape[1])[:ns[0]]]
    u = model(batch[base_idxs[0]])
    v = model(batch[base_idxs[1]])
    pos_score = model.score(u, v)
    loss += pi[0] * F.cross_entropy(pos_score, torch.empty(ns[0], dtype=torch.long).fill_(0).to(device))

    base_idxs = pos_idxs[:, torch.randperm(pos_idxs.shape[1])[:ns[1]]]
    dist_idxs = torch.tensor([rnd_idx_without(0, 2*n_b, base_idxs[:,j]) for j in range(ns[1])])
    u = model(batch[base_idxs[0]])
    v = model(batch[dist_idxs])
    neg_score = model.score(u, v)
    loss += pi[1] * F.cross_entropy(neg_score, torch.empty(ns[1], dtype=torch.long).fill_(1).to(device))
    
    if acc:
        # Compute accuracy
        predictions = torch.cat([pos_score.argmax(dim=1), neg_score.argmax(dim=1)], dim=0)
        labels = torch.cat([torch.zeros(pos_score.shape[0]), torch.ones(neg_score.shape[0])], dim=0).to(device)
        accuracy = (predictions == labels).float().mean()
        accs = {'accuracy': accuracy}
        return accs
    
    return loss

def prob_loss(model, batch1, batch2, temperature, device, acc=False):
    # Set up parameters
    num_modalities = len(batch1) if not torch.is_tensor(batch1) else 1
    if num_modalities == 1:
        return _prob_loss_single_modality(model, batch1, batch2, temperature, device, acc)
    n_b = batch1[0].shape[0] if num_modalities > 1 else batch1.shape[0]
    to_disturb = generate_binary_combinations(num_modalities)
    n = n_b*len(to_disturb)
    pi = create_distribution(len(to_disturb))
    ns = [int(n*p) for p in pi]

    pos_idxs = torch.cat([torch.stack([torch.arange(0, n_b), torch.arange(n_b, 2*n_b)], dim=0), torch.stack([torch.arange(n_b, 2*n_b), torch.arange(0, n_b)], dim=0)], dim=1)
    batch = [torch.cat([batch1[i], batch2[i]], dim=0) for i in range(num_modalities)]
    losses = []
    scores = []
    for i, disturb in enumerate(to_disturb):
        # Select ns[i] elements from pos_idxs at random
        base_idxs = pos_idxs[:, torch.randperm(pos_idxs.shape[1])[:ns[i]]]
        # Get anchor representations
        u = model([batch[j][base_idxs[0]] for j in range(num_modalities)])
        # Genearate indexes for each modality based on what needs to be disturbed
        idxs = []
        for j in range(num_modalities):
            dist_idxs = None
            if disturb[j]:
                if dist_idxs is None:
                    dist_idxs = torch.tensor([rnd_idx_without(0, 2*n_b, base_idxs[:,j]) for j in range(ns[i])])
                idxs.append(dist_idxs)
            else:
                idxs.append(base_idxs[1])
        # Get representations to compare to anchors
        v = model([batch[j][idxs[j]] for j in range(num_modalities)])
        # Get scores
        score = model.score(u, v)
        if not acc:
            # Get loss
            losses.append(pi[i] * F.cross_entropy(score, torch.empty(ns[i], dtype=torch.long).fill_(i).to(device)))
        else:
            scores.append(score)
    
    if acc:
        # Compute accuracy
        predictions = torch.cat([scores.argmax(dim=1) for scores in scores], dim=0)
        labels = torch.cat([i*torch.ones(s.shape[0]) for i, s in enumerate(scores)], dim=0).to(device)
        accuracy = (predictions == labels).float().mean()
        accs = {'accuracy': accuracy}
        return accs
    
    return sum(losses)

def decomposed_loss(model, batch1, batch2, temperature, device, acc=False):
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

        accs = {}
        accuracy = []
        for i in range(len(to_disturb[0])):
            l = [torch.empty(s.shape[0], dtype=torch.long).fill_(to_disturb[j][i]).to(device) for j, s in enumerate(scores)]
            acc_i = (predictions == torch.cat(l, dim=0)).float().mean()
            accs[f'class_{i}'] = acc_i
            accuracy.append(acc_i)
        accs['accuracy'] = sum(accuracy) / len(accuracy)
        return accs

    # We then compute the cross entropy loss between the scores and the correct logits
    losses = []
    for i, s in enumerate(scores):
        for j, d in enumerate(to_disturb[i]):
            losses.append(F.binary_cross_entropy_with_logits(s[:,j], torch.empty(s.shape[0], dtype=torch.float).fill_(d).to(device), reduction='mean'))
    # We then sum the losses and return them
    loss = sum(losses) / len(losses)
    return loss
   

def SimCLR_loss(model, batch1, batch2, device, acc=False, temperature=1, **kwargs):
    batch_size = batch1[0].shape[0] if isinstance(batch1, (tuple, list)) else batch1.shape[0]
    LARGE_NUM = 1e9

    batch1 = [b.to(device) if torch.is_tensor(b) else b for b in batch1] if not torch.is_tensor(batch1) else batch1.to(device)
    y1 = model(batch1)

    batch2 = [b.to(device) if torch.is_tensor(b) else b for b in batch2] if not torch.is_tensor(batch2) else batch2.to(device)
    y2 =  model(batch2)

    masks = F.one_hot(torch.arange(batch_size), num_classes=batch_size).float().to(device)

    logits_aa = torch.matmul(y1, y1.t()) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_ab = torch.matmul(y1, y2.t()) / temperature  
    loss_a = F.cross_entropy(
        torch.cat([logits_ab, logits_aa], dim=1),
        torch.arange(batch_size).to(device)
    )

    logits_bb = torch.matmul(y2, y2.t()) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ba = torch.matmul(y2, y1.t()) / temperature
    loss_b = F.cross_entropy(
        torch.cat([logits_ba, logits_bb], dim=1),
        torch.arange(batch_size).to(device)
    )
    loss = loss_a + loss_b

    if acc==True:
        labels = F.one_hot(torch.arange(batch_size), num_classes=batch_size * 2).float().to(device)
        labels = labels.argmax(dim=1)
        predictions = logits_ab.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        accs = {'accuracy': accuracy}
        return accs
    
    return loss

def get_train_accuracy(model, batch1, batch2, est, device):
    if est == 'info_critic':
        accs = info_critic(model, batch1, batch2, device, acc=True)
    elif est == 'info_critic_plus':
        accs = info_critic_plus(model, batch1, batch2, device, acc=True)
    elif est == 'prob_loss':
        accs = prob_loss(model, batch1, batch2, device, acc=True)
    elif est == 'decomposed_loss':
        accs = decomposed_loss(model, batch1, batch2, device, acc=True)
    elif est == 'SimCLR':
        accs = SimCLR_loss(model, batch1, batch2, device, acc=True)
    else:
        raise Exception('Unknown estimation method')
    return accs