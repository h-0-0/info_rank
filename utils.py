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

def generate_binary_combinations(n):
    if n <= 0:
        return []
    
    # Using list comprehension and itertools.product
    import itertools
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
