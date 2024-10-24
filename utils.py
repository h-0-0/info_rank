import torch

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
    elif modality == 'image_ft+audio_ft+text':
        modality0, modality1, modality2, labels = batch
        modality0, modality1, modality2, labels = modality0.to(device), modality1.to(device), modality2.to(device), labels.to(device)
        batch = (modality0, modality1, modality2)
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
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes).to(device)

        metrics['acc'] = accuracy
        metrics['IoU'] = jaccard
    elif modality == 'image_ft+audio_ft+text':
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

from model import FusionModel, ImageModel, AudioModel, ResNet101, ResNet50, FullConvNet, MosiFusion, MoseiFusion
from misa import MISA
def get_model(model, output_dim):
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
        model = MISA('mosi', output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
    elif model == "MoseiMISA":
        model = MISA('mosei', output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
    else:
        raise ValueError("Invalid model")
    return model, modality

from model import LinearClassifier, SegClassifier, Regression, FCN_SegHead
def get_classifier_criterion(model_name, output_dim, benchmark):
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
    elif model_name in ["MosiFusion", "MoseiFusion", "MosiMISA", "MoseiMISA", "MosiFusionAttention", "MoseiFusionAttention"]:
        classifier = Regression(input_dim=output_dim)
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
    else:
        raise ValueError("Invalid model name")
    return classifier, criterion