import torch
import os 
import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def dict_to_ls(**kwargs):
    formatted_args = []

    for name, value in kwargs.items():
        formatted_arg = f"--{name}={value}"
        formatted_args.append(formatted_arg)

    return formatted_args

def ls_to_dict(ls):
    d = {}
    for item in ls:
        key, value = item.split('=')
        key = key[2:]
        # Attempt to convert value to int or float
        if ('.' in value) or ('e' in value):
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            try:
                value = int(value)
            except ValueError:
                pass
        d[key] = value
    return d

import itertools
def generate_binary_combinations(n):
    if n <= 0:
        return []
    return list(itertools.product([0, 1], repeat=n))

def swap_halves(tensor, dim=0):
    # Split the tensor into two equal halves along the specified dimension
    first_half, second_half = torch.chunk(tensor, 2, dim=dim)
    
    # Concatenate the halves in reverse order
    swapped_tensor = torch.cat((second_half, first_half), dim=dim)
    
    return swapped_tensor

def dual_batch_indice_permutation(n):
    """
    Returns a tensor of indices taking values in [0,..., 2n-1],
    such that i-th element doesn't contain i%n or (i+n)%n.
    Note that the same indice could appear twice in the resulting permutation if the indice at element i the value is i%n or (i+n)%n.
    """
    # Randomly tensor of random permutation of indices
    indices = torch.randperm(2*n)
    # Check that i-th element doesn't contain i%n or (i+n)%n
    # if it does, then sample from [1, ..., i%n -1, i%n +1, ..., (i+n)%n -1, (i+n)%n +1, ...,  n-1]
    for i in range(2*n):
        if indices[i]%n  == i%n :
            if i%n == 0:
                indices[i] = torch.randint(1, n, (1,)) * torch.randint(1, 3, (1,))
            elif (i%n) == n-1:
                indices[i] = torch.randint(0, n-1, (1,)) * torch.randint(1, 3, (1,))
            else:
                possible_indices = torch.cat([torch.arange(1, i%n), torch.arange((i%n)+1, (i%n)+n), torch.arange((i%n)+n+1, 2*n)])
                indices[i] = possible_indices[torch.randint(0, possible_indices.shape[0], (1,))]
    return indices

def create_distribution(n):
    if n < 2:
        raise ValueError("n must be at least 2")
    
    distribution = [0.1] + [0.9 / (n - 1)] * (n - 1)
    return distribution

def rnd_idx_without(start, end, idxs):
    x = torch.tensor([i for i in range(start, end) if i not in idxs])
    return x[torch.randint(0, len(x), (1,)).item()]

import torch.nn as nn
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class FusedOpt():
    """
    Use this class when using a fused optimizer so that compatible.
    Refer to: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
    """
    def __init__(self, opt_dict) -> None:
        self.opt_dict = opt_dict

    def step(self):
        "Dont do anything as everything updated in the hook"
        pass

    def zero_grad(self):
        "Zero the gradients of all the parameters."
        pass
        # for opt in self.opt_dict.values():
        #     opt.zero_grad()
    

from torch import optim
def get_optimizer(model, learning_rate, fuse_opt=False):
    if fuse_opt:        
        print("Fusing")
        optimizer_dict = {p: optim.SGD([p], foreach=False, lr=learning_rate) for p in model.parameters()}
        # Define our hook, which will call the optimizer ``step()`` and ``zero_grad()``
        def optimizer_hook(parameter) -> None:
            optimizer_dict[parameter].step()
            optimizer_dict[parameter].zero_grad()

        # Register the hook onto every parameter
        for p in model.parameters():
            p.register_post_accumulate_grad_hook(optimizer_hook)
        opt = FusedOpt(optimizer_dict)
    else:
        opt = optim.SGD(model.parameters(), lr=learning_rate)
    return opt

import psutil
def log_memory(writer, step):
    # Log CPU memory
    cpu_memory_used = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
    writer.add_scalar('Memory/CPU_Used_MB', cpu_memory_used, step)

    # Log GPU memory if available
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # Convert to MB
        writer.add_scalar('Memory/GPU_Allocated_MB', gpu_memory_allocated, step)
        writer.add_scalar('Memory/GPU_Reserved_MB', gpu_memory_reserved, step)

def get_batch_labels(modality, batch, device):
    if modality in ['image+audio', 'image', 'audio', 'image+depth']:
        modality0, modality1, labels = batch
        if modality in ['image+audio', 'image+depth']:
            modality0, modality1, labels = modality0.to(device), modality1.to(device), labels.to(device)
            batch = (modality0, modality1)
        elif modality == 'image':
            modality0, labels = modality0.to(device), labels.to(device)
            batch = modality0
        elif modality == 'audio':
            modality1, labels = modality1.to(device), labels.to(device)
            batch = modality1
    elif modality in ['image_ft+audio_ft+text', 'image_ft+text']:
        modality0, modality1, modality2, labels = batch
        modality0, modality1, modality2, labels = modality0.to(device), modality1.to(device), modality2.to(device), labels.to(device)
        batch = (modality0, modality1, modality2)
    elif modality == 'image_ft+audio_ft+text_bert':
        # sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask = batch
        # sentences, visual, acoustic, bert_sentences, bert_sentence_types, bert_sentence_att_mask, lengths, labels = sentences.to(device), visual.to(device), acoustic.to(device), bert_sentences.to(device), bert_sentence_types.to(device), bert_sentence_att_mask.to(device), lengths.to('cpu'), labels.to(device)
        # batch = (visual, acoustic, sentences, bert_sentences, bert_sentence_types, bert_sentence_att_mask, lengths)
        visual, acoustic, sentences, labels = batch
        visual, acoustic, sentences, labels = visual.to(device), acoustic.to(device), sentences, labels.to(device)
        batch = (visual, acoustic, sentences)
    else:
        raise ValueError("Invalid modality")
    return batch, labels

import torchmetrics

class CustomBinaryAccuracy(torchmetrics.Metric):
    def __init__(self, threshold=0, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Convert scalar predictions to binary, using a soft threshold
        preds_binary = (preds >= self.threshold).long()
        target_binary = (target >= self.threshold).long()

        # Calculate correct predictions
        correct = torch.eq(preds_binary, target_binary).sum()
        
        self.correct += correct
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
    
class CustomStrictBinaryAccuracy(torchmetrics.Metric):
    def __init__(self, threshold=0, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Get rid of targets that are exactly 0
        mask = target != 0
        preds = preds[mask]
        target = target[mask]
        # Convert scalar predictions to binary, here we use a strict threshold
        preds_binary = (preds > self.threshold).long()
        target_binary = (target > self.threshold).long()

        # Calculate correct predictions
        correct = torch.eq(preds_binary, target_binary).sum()
        
        self.correct += correct
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

class CustomSevenAccuracy(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Convert scalar predictions to class in [0, 6]
        preds_7 = torch.round(preds) +3
        target_7 = torch.round(target) +3

        # Calculate correct predictions
        correct = torch.eq(preds_7, target_7).sum()
        
        self.correct += correct
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

from torchmetrics.functional import f1_score
class CustomF1Score(torchmetrics.Metric):
    def __init__(self, threshold=0, average='macro', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.average = average
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds = torch.round(preds) +3 #TODO: remove
        # target = torch.round(target) +3 #TODO: remove
        preds = (preds >= self.threshold).long()
        target = (target >= self.threshold).long()
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)
        return f1_score(preds, target, average=self.average, task='binary')
    
def get_torchmetrics(modality, num_classes, device):
    metrics = {}
    if modality in ['image', 'audio', 'image+audio']:
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        
        metrics['acc'] = accuracy
    elif modality == 'image+depth':
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=-1).to(device)
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=-1, average='macro', validate_args=True).to(device) # TODO: change validate_args to False

        metrics['acc'] = accuracy
        metrics['IoU'] = jaccard
    elif modality in ['image_ft+audio_ft+text', 'image_ft+text', 'image_ft+audio_ft+text_bert']:
        mse = torchmetrics.MeanSquaredError().to(device)
        seven_accuracy = CustomSevenAccuracy().to(device)
        binary_accuracy = CustomBinaryAccuracy().to(device)
        strict_binary_accuracy = CustomStrictBinaryAccuracy().to(device)
        f1 = CustomF1Score().to(device)

        metrics['mse'] = mse
        metrics['7_acc'] = seven_accuracy
        metrics['f1'] = f1
        metrics['binary_acc'] = binary_accuracy
        metrics['strict_binary_acc'] = strict_binary_accuracy
    else:
        raise ValueError("Invalid modality")
    return metrics

from model import FusionModel, ImageModel, AudioModel, ResNet101, ResNet50, FullConvNet, MosiFusion, MoseiFusion, ESANet_18, MosiTransformer
from misa_model import MISA
from mult_model import MULTModel
def get_model(model, benchmark, output_dim):
    if model == "FusionModel":
        model = FusionModel(output_dim=output_dim)
        modality = 'image+audio'
    elif model == "ImageOnly":
        model = ImageModel(output_dim=output_dim)
        modality = 'image'
    elif model == "AudioOnly":
        model = AudioModel(output_dim=output_dim)
        modality = 'audio'
    elif model == "ResNet101":
        model = ResNet101(output_dim=output_dim)
        modality = 'image+depth'
    elif model == "ResNet50":
        model = ResNet50(output_dim=output_dim)
        modality = 'image+depth'
    elif model == "FCN50":
        model = FullConvNet(resnet='resnet50')
        modality = 'image+depth'
    elif model == "FCN101":
        model = FullConvNet(resnet='resnet101')
        modality = 'image+depth'
    elif model == "ESANet_18":
        model = ESANet_18()
        modality = 'image+depth'
    elif model == "MosiFusion":
        model = MosiFusion(output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
    elif model == "MosiFusionAttention":
        model = MosiFusion(output_dim=output_dim, attention=True)
        modality = 'image_ft+audio_ft+text'
    elif model == "MoseiFusion":
        model = MoseiFusion(output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
    elif model == "MoseiFusionAttention":
        model = MoseiFusion(output_dim=output_dim, attention=True)
        modality = 'image_ft+audio_ft+text'
    elif model == "MosiMISA":
        model = MISA('mosi', output_dim=128, use_bert=True)
        if benchmark == 'mosi_bert':
            modality = 'image_ft+audio_ft+text_bert'
        elif benchmark == 'mosi':
            modality = 'image_ft+audio_ft+text'
        else: 
            raise ValueError("Invalid benchmark for MosiMISA")
    elif model == "MoseiMISA":
        model = MISA('mosei', output_dim=128, use_bert=True)
        if benchmark == 'mosei_bert':
            modality = 'image_ft+audio_ft+text_bert'
        elif benchmark == 'mosei':
            modality = 'image_ft+audio_ft+text'
        else: 
            raise ValueError("Invalid benchmark for MoseiMISA")
    elif model == "MosiMULT":
        model = MULTModel('mosi', output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
    elif model == "MoseiMULT":
        model = MULTModel('mosei', output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
    elif model == "MosiTransformer":
        model = MosiTransformer()
        modality = 'image_ft+audio_ft+text'
    elif model == "MosiTransformer_NoAudio":
        model = MosiTransformer(no_audio=True)
        modality = 'image_ft+text'
    else:
        raise ValueError("Invalid model")
    return model, modality

import numpy as np
class SegCrossEntropyLoss(nn.Module):
    def __init__(self, device, weight):
        super(SegCrossEntropyLoss, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='none',
            ignore_index=-1
        )
        self.ce_loss.to(device)

    def forward(self, inputs_scales, targets_scales):
        # losses = []
        number_of_pixels_per_class = torch.stack([torch.bincount(targets_scales[i].flatten().type(self.dtype), minlength=self.num_classes) for i in range(targets_scales.shape[0])])
        divisor_weighted_pixel_sum = torch.sum(number_of_pixels_per_class[:,1:] * self.weight, dim=[1])   # without void
        targets_scales -= 1
        loss_all = self.ce_loss(inputs_scales, targets_scales.long())
        loss = torch.div(torch.sum(loss_all, dim=[1,2]), divisor_weighted_pixel_sum)
        loss = torch.mean(loss)
        return loss
        # for inputs, targets in zip(inputs_scales, targets_scales):
        #     # mask = targets > 0
        #     targets_m = targets.clone()
        #     targets_m -= 1
        #     print(inputs.shape, targets_m.shape)
        #     loss_all = self.ce_loss(inputs, targets_m.long())

        #     number_of_pixels_per_class = \
        #         torch.bincount(targets.flatten().type(self.dtype),
        #                        minlength=self.num_classes)
        #     divisor_weighted_pixel_sum = \
        #         torch.sum(number_of_pixels_per_class[1:] * self.weight)   # without void
        #     losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
        #     # losses.append(torch.sum(loss_all) / torch.sum(mask.float()))

        # return losses

def compute_class_weights(data_loader, weight_mode='median_frequency', c=1.02, n_classes_with_void=13, source_path='data', split='train'):
        assert weight_mode in ['median_frequency', 'logarithmic', 'linear']
        n_classes_without_void = n_classes_with_void - 1
        # build filename
        class_weighting_filepath = os.path.join(
            source_path, f'weighting_{weight_mode}_'
                              f'1+{n_classes_without_void}')
        if weight_mode == 'logarithmic':
            class_weighting_filepath += f'_c={c}'

        class_weighting_filepath += f'_{split}.pickle'

        if os.path.exists(class_weighting_filepath):
            class_weighting = pickle.load(open(class_weighting_filepath, 'rb'))
            print(f'Using {class_weighting_filepath} as class weighting')
            return class_weighting

        print('Compute class weights')

        n_pixels_per_class = np.zeros(n_classes_with_void)
        n_image_pixels_with_class = np.zeros(n_classes_with_void)
        for batch in data_loader:
            _ , _, label = batch
            _, h, w = label.shape
            current_dist = np.bincount(label.flatten(),
                                       minlength=n_classes_with_void)
            n_pixels_per_class += current_dist

            # For median frequency we need the pixel sum of the images where
            # the specific class is present. (It only matters if the class is
            # present in the image and not how many pixels it occupies.)
            class_in_image = current_dist > 0
            n_image_pixels_with_class += class_in_image * h * w

        # remove void
        n_pixels_per_class = n_pixels_per_class[1:]
        n_image_pixels_with_class = n_image_pixels_with_class[1:]

        if weight_mode == 'linear':
            class_weighting = n_pixels_per_class

        elif weight_mode == 'median_frequency':
            frequency = n_pixels_per_class / n_image_pixels_with_class
            class_weighting = np.median(frequency) / frequency

        elif weight_mode == 'logarithmic':
            probabilities = n_pixels_per_class / np.sum(n_pixels_per_class)
            class_weighting = 1 / np.log(c + probabilities)

        if np.isnan(np.sum(class_weighting)):
            print(f"n_pixels_per_class: {n_pixels_per_class}")
            print(f"n_image_pixels_with_class: {n_image_pixels_with_class}")
            print(f"class_weighting: {class_weighting}")
            raise ValueError('class weighting contains NaNs')

        with open(class_weighting_filepath, 'wb') as f:
            pickle.dump(class_weighting, f)
        print(f'Saved class weights under {class_weighting_filepath}.')
        return class_weighting    

from model import LinearClassifier, SegClassifier, Regression, FCN_SegHead, ESANet_18_Decoder
def get_classifier_criterion(model_name, output_dim, benchmark, train_loader, device):
    if model_name in ["FusionModel", "ImageOnly", "AudioOnly"]:
        num_classes = 10
        classifier = LinearClassifier(output_dim, num_classes)
        criterion = nn.CrossEntropyLoss()
    elif model_name in ["ResNet101", "ResNet50"]:
        if benchmark == 'nyu_v2_13':
            num_classes = 14
        elif benchmark == 'nyu_v2_40':
            num_classes = 41
        else:
            raise ValueError("Invalid benchmark and model combination")
        classifier = SegClassifier(num_classes)
        criterion = nn.CrossEntropyLoss()
    elif model_name in ["MosiFusion", "MoseiFusion", "MosiMISA", "MoseiMISA", "MosiFusionAttention", "MoseiFusionAttention", "MosiMULT", "MoseiMULT", "MosiTransformer", "MosiTransformer_NoAudio"]:
        if model_name in ["MosiMISA", "MoseiMISA"]:
            out_dropout=0.5
        else: 
            out_dropout=0.0
        classifier = Regression(input_dim=output_dim, out_dropout=out_dropout)
        if model_name in ["MosiMULT", "MoseiMULT"]:
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
    elif model_name == "FCN50":
        if benchmark == 'nyu_v2_13':
            num_classes = 14
        elif benchmark == 'nyu_v2_40':
            num_classes = 41
        classifier = FCN_SegHead(num_classes, resnet='resnet50')
        criterion = nn.CrossEntropyLoss()
    elif model_name == "FCN101":
        if benchmark == 'nyu_v2_13':
            num_classes = 14
        elif benchmark == 'nyu_v2_40':
            num_classes = 41
        classifier = FCN_SegHead(num_classes, resnet='resnet101')
        criterion = nn.CrossEntropyLoss()
    elif model_name == "ESANet_18":
        if benchmark == 'nyu_v2_13':
            num_classes = 13
        elif benchmark == 'nyu_v2_40':
            num_classes = 40
        classifier = ESANet_18_Decoder(num_classes)
        class_weighting = compute_class_weights(train_loader, weight_mode='median_frequency', n_classes_with_void=num_classes+1)
        print(f"Class weighting: {class_weighting}")
        criterion = SegCrossEntropyLoss(device, class_weighting)
    else:
        raise ValueError("Invalid model name")
    return classifier, criterion