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