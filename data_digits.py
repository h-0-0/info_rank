# Please excuse how janky this script is.
import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.transforms import v2
import torchvision.transforms.functional as F
from typing import Tuple
import librosa
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, Subset, DataLoader

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def order_reduce_mnist(data, target, train=True):
    # Order the data according to the targets
    order = np.argsort(target)
    data = data[order]
    target = target[order]
    # Remove same number of samples from each class so that we have a balanced dataset
    if train == True:
        samples_per_class = [2620, 2603, 2613, 2598, 2621, 2614, 2616, 2571, 2590, 2554]
    else:
        samples_per_class = [380,  397,  387,  402,  379,  386,  384,  429,  410,  446]
    for i in range(10):
        idx = np.where(target == i)[0]
        idx = idx[samples_per_class[i]:]
        data = np.delete(data, idx, axis=0)
        target = np.delete(target, idx, axis=0)
    return data, target

def get_ImageMNIST(sigma=0) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:     
    """ 
    Downloads the MNIST dataset (https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py).
    Returns the train and test set, with normalization transform applied.

    Returns:
        - train_set (torch.utils.data.Dataset): The training set
        - test_set (torch.utils.data.Dataset): The test set
    """   
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    # Download and load MNIST dataset with labels as integers (0-9)
    train_set = datasets.MNIST(root=os.path.join('data', 'mnist_image_train'), train=True, download=True)
    test_set = datasets.MNIST(root=os.path.join('data', 'mnist_image_test'), train=False, download=True)
    
    # Keep only 26000 training samples (equal across classes)
    train_data, train_targets = order_reduce_mnist(train_set.data.numpy(), train_set.targets.numpy(), train=True)
    # Keep only 4000 test samples (equal across classes)
    test_data, test_targets = order_reduce_mnist(test_set.data.numpy(), test_set.targets.numpy(), train=False)
    # Add channel dimension
    train_data = train_data[:,:,:, np.newaxis]
    test_data = test_data[:,:,:, np.newaxis]
    # Apply transform to training and test set
    train_data = torch.stack([transform(d) for d in train_data])
    if sigma > 0:
        train_data += torch.randn(train_data.shape) * sigma
    train_targets = torch.tensor(train_targets, dtype=torch.long)
    test_data = torch.stack([transform(d) for d in test_data])
    if sigma > 0:
        test_data += torch.randn(test_data.shape) * sigma
    test_targets = torch.tensor(test_targets, dtype=torch.long)
    
    return train_data, train_targets, test_data, test_targets

def download_AudioMNIST(viz = True, cleanup=True):
    """
    Downloads the AudioMNIST dataset (https://github.com/soerenab/AudioMNIST).

    Args:
        - viz (bool): Whether to create plots of the audio data
        - cleanup (bool): Whether to delete the cloned directory of the original data after processing
    
    """
    # Download the dataset if it doesn't exist
    if not os.path.exists(os.path.join('data','AudioMNIST')):
        os.system('git clone https://github.com/soerenab/AudioMNIST.git '+ os.path.join('data','AudioMNIST'))
    # Check directory for viz exists
    if viz and not os.path.exists(os.path.join('viz','written_spoken_digits','AudioMNIST')):
        os.makedirs(os.path.join('viz', 'written_spoken_digits', 'AudioMNIST'))
    # Create numpy array of the data
    audio_data = []
    label_data = []
    for digit in range(0,10):
        rndm_speaker = np.random.randint(0,60)
        rndm_index = np.random.randint(0,50)
        for speaker in range(1,61):
            for index in range(0,50):
                if speaker<10:
                    file = os.path.join("data", "AudioMNIST", "data", f"0{speaker}", f"{digit}_0{speaker}_{index}.wav")
                else:
                    file = os.path.join("data", "AudioMNIST", "data", f"{speaker}", f"{digit}_{speaker}_{index}.wav")
                audio, sample_rate = librosa.load(file)
                if viz and speaker==rndm_speaker and index==rndm_index:
                    plt.figure(figsize=(10, 4))
                    plt.plot(audio)
                    plt.title(f"Digit {digit} - Speaker {speaker} - Index {index}")
                    plt.savefig(os.path.join("viz", 'written_spoken_digits', "AudioMNIST", f"digit_{digit}_speaker_{speaker}_{index}.png"))
                    plt.close()
                
                    os.system(f"cp {file} {os.path.join('viz','written_spoken_digits', 'AudioMNIST', f'digit_{digit}_speaker_{speaker}_{index}.wav')}")
                n_mfcc = 40
                mfcc = librosa.feature.mfcc(y=audio,sr=sample_rate, n_mfcc=n_mfcc)
                mfcc = padding(mfcc, n_mfcc, 44)

                if viz and speaker==rndm_speaker and index==rndm_index:
                    plt.figure(figsize=(10, 4))
                    plt.imshow(mfcc, aspect='auto', origin='lower')
                    plt.colorbar()
                    plt.title(f"Digit {digit} - Speaker {speaker} - Index {index}")
                    plt.savefig(os.path.join("viz", 'written_spoken_digits', "AudioMNIST", f"digit_{digit}_speaker_{speaker}_{index}_mfcc.png"))
                    plt.close()

                audio_data.append(mfcc)
                label_data.append(digit)
    audio_data = np.stack(audio_data)          
    label_data = np.stack(label_data)
    np.save('data/mnist_audio_data', audio_data, allow_pickle =False)
    np.save('data/mnist_audio_labels', label_data, allow_pickle =False)
    if cleanup:
        os.system('rm -rf data/AudioMNIST/')

def min_max_normalization(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
    return normalized_tensor

def get_AudioMNIST(sigma=0) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Checks if the AudioMNIST dataset (https://github.com/soerenab/AudioMNIST) is downloaded and processed, returns it if it is.
    If it isn't then will download it, split into a train and test set and for each; create two tensors, one for the audio and one for the labels, process the audio into MFCCs and save them as a pytorch dataset.

    Args:
        - sigma (float): The variance of the (zero mean) gaussian noise to add to the audio

    Returns:
        - train_dataset (torch.utils.data.Dataset): The training dataset
        - test_dataset (torch.utils.data.Dataset): The test dataset

    """
    np.random.seed(42)
    # Check if np files exist and load them if they do
    if os.path.exists('data/mnist_audio_data.npy') and os.path.exists('data/mnist_audio_labels.npy'):
        audio_data = np.load('data/mnist_audio_data.npy')
        audio_labels = np.load('data/mnist_audio_labels.npy')
    else:
        download_AudioMNIST()
        audio_data = np.load('data/mnist_audio_data.npy')
        audio_labels = np.load('data/mnist_audio_labels.npy')
    # Split the data into train and test sets
    indices = np.random.permutation(audio_data.shape[0])
    train_indices = indices[:30000-4000]
    test_indices = indices[30000-4000:]
    train_data = audio_data[train_indices]
    train_labels = audio_labels[train_indices]
    test_data = audio_data[test_indices]
    test_labels = audio_labels[test_indices]
    # Add channel dimension and order the data arrays according to the labels
    sorted_train_indices = np.argsort(train_labels)
    train_data = train_data[sorted_train_indices][:,:,:, np.newaxis]
    train_labels = train_labels[sorted_train_indices]

    sorted_test_indices = np.argsort(test_labels)
    test_data = test_data[sorted_test_indices][:,:,:, np.newaxis]
    test_labels = test_labels[sorted_test_indices]
    
    # Convert to tensors
    # trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) #TODO Remove
    trans = transforms.Compose([transforms.ToTensor()])
    
    # train_data = min_max_normalization(train_data)
    train_data = torch.stack([trans(d) for d in train_data])
    if sigma > 0:
        train_data += torch.randn(train_data.shape) * sigma

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    # test_data = min_max_normalization(test_data)
    test_data = torch.stack([trans(d) for d in test_data])
    if sigma > 0:
        test_data += torch.randn(test_data.shape) * sigma
    
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_data, train_labels, test_data, test_labels


def digits_get_data_loaders(batch_size):
    """
    Returns the train and test data loaders for the bi-modal dataset consisting of spoken mnist (audio) and written mnist (images).
    The data is split into a train and test set, with each set containing 30000 and 4000 samples respectfully.

    Args:
        - batch_size (int): The batch size

    Returns:
        - train_loader (torch.utils.data.DataLoader): The training data loader
        - test_loader (torch.utils.data.DataLoader): The test data loader

    """
    # Check if datasets exist and load them if they do
    # if os.path.exists(os.path.join('data','digits_train_data.pt')) and os.path.exists(os.path.join('data','digits_test_data.pt')):
    #     train_data = torch.load('data/digits_train_data.pt')
    #     test_data = torch.load('data/digits_test_data.pt')
    # else:
    audio_train_data, audio_train_labels, audio_test_data, audio_test_labels = get_AudioMNIST()
    images_train_data, images_train_labels, images_test_data, images_test_labels = get_ImageMNIST()
    assert (audio_train_labels == images_train_labels).all()
    assert (audio_test_labels == images_test_labels).all()
    # Combine training sets so that we match images and audio samples according to label
    train_data = torch.utils.data.TensorDataset(images_train_data, audio_train_data, images_train_labels)
    # Combine test sets so that we match images and audio samples according to label
    test_data = torch.utils.data.TensorDataset(images_test_data, audio_test_data, images_test_labels)
    # Save the datasets
    # torch.save(train_data, 'data/digits_train_data.pt')
    # torch.save(test_data, 'data/digits_test_data.pt')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def digits_get_data_loaders_weak_modality(batch_size, sigma, weaken_audio=False, weaken_image=False):
    """
    Returns the train and test data loaders for the bi-modal dataset consisting of spoken mnist (audio) and written mnist (images).
    The data is split into a train and test set, with each set containing 30000 and 4000 samples respectfully.
    If weaken_audio is True then the audio data will be weakened by adding zero mean variance sigma gaussian noise to it.
    If weaken_image is True then the image data will be weakened by adding zero mean variance sigma gaussian noise to it.

    Args:
        - batch_size (int): The batch size
        - sigma (float): The variance of the gaussian noise to add to the data
        - weaken_audio (bool): Whether to weaken the audio data
        - weaken_image (bool): Whether to weaken the image data

    Returns:
        - train_loader (torch.utils.data.DataLoader): The training data loader
        - test_loader (torch.utils.data.DataLoader): The test data loader

    """  
    if (not weaken_audio) and (not weaken_image):
        print("No weakening applied")
        return digits_get_data_loaders(batch_size)
    # Check if datasets exist and load them if they do
    if not (sigma > 0):
        raise ValueError("Sigma must be greater than 0 if weakening is applied")
    audio_sigma = sigma if weaken_audio else 0
    image_sigma = sigma if weaken_image else 0

    audio_train_data, audio_train_labels, audio_test_data, audio_test_labels = get_AudioMNIST(sigma=audio_sigma)
    images_train_data, images_train_labels, images_test_data, images_test_labels = get_ImageMNIST(sigma=image_sigma)
    assert (audio_train_labels == images_train_labels).all()
    assert (audio_test_labels == images_test_labels).all()
    # Combine training sets so that we match images and audio samples according to label
    train_data = torch.utils.data.TensorDataset(images_train_data, audio_train_data, images_train_labels)
    # Combine test sets so that we match images and audio samples according to label
    test_data = torch.utils.data.TensorDataset(images_test_data, audio_test_data, images_test_labels)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def digits_get_data_loaders_noisy_pairing(batch_size, p):
    """
    Returns the train and test data loaders for the bi-modal dataset consisting of spoken mnist (audio) and written mnist (images).
    The data is split into a train and test set, with each set containing 30000 and 4000 samples respectfully.

    Args:
        - batch_size (int): The batch size
        - p (float): Probability of an incorrect pairing between modalities
        - weaken_audio (bool): Whether to weaken the audio data
        - weaken_image (bool): Whether to weaken the image data

    Returns:
        - train_loader (torch.utils.data.DataLoader): The training data loader
        - test_loader (torch.utils.data.DataLoader): The test data loader

    """  
    if not (p > 0):
        raise ValueError("p must be greater than 0 if weakening is applied")

    audio_train_data, audio_train_labels, audio_test_data, audio_test_labels = get_AudioMNIST()
    images_train_data, images_train_labels, images_test_data, images_test_labels = get_ImageMNIST()
    assert (audio_train_labels == images_train_labels).all()
    assert (audio_test_labels == images_test_labels).all()
    # With probability p change which image an audio pair is associated to and vice-versa
    for i in range(len(audio_train_labels)):
        if np.random.rand() < p:
            swap_idx = np.random.randint(0, len(audio_train_labels))
            audio_train_data[i], audio_train_data[swap_idx] = audio_train_data[swap_idx], audio_train_data[i]
    for i in range(len(images_train_labels)):
        if np.random.rand() < p:
            swap_idx = np.random.randint(0, len(images_train_labels))
            images_train_data[i], images_train_data[swap_idx] = images_train_data[swap_idx], images_train_data[i]
    # Combine training sets so that we match images and audio samples according to label
    train_data = torch.utils.data.TensorDataset(images_train_data, audio_train_data, images_train_labels)
    # Combine test sets so that we match images and audio samples according to label
    test_data = torch.utils.data.TensorDataset(images_test_data, audio_test_data, images_test_labels)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

if __name__ == '__main__':
    """
    Following code is for testing the data fetching and pre-processing functions, simply run this file to test them.
    Produces some visualizations of the data and saves some of the audio in listenable format.
    """
    print("------ Find visualisations in the /viz directory ------")
    n_b = 4
    train_loader, test_loader = digits_get_data_loaders(4)
    image_batch, audio_batch, label_batch = next(iter(train_loader))
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Audio batch shape: {audio_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")

    # Create plot of subplots where each column shows the image and mfcc audio with title telling us the label for one sample from the batch
    fig, axes = plt.subplots(2, n_b, figsize=(10, 5))
    for i in range(n_b):
        axes[0, i].imshow(image_batch[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Label: {label_batch[i]}")
        axes[1, i].imshow(audio_batch[i].squeeze(), cmap='gray')
    if not os.path.exists('viz/written_spoken_digits'):
        os.makedirs('viz/written_spoken_digits')
    plt.savefig('viz/written_spoken_digits/batch.png')
    plt.close()

    # Plot bar chart of number of samples in each class
    plt.figure()
    plt.bar(range(10), [len(train_loader.dataset[i]) for i in range(10)])
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Number of samples in each class')
    plt.xticks(range(10))
    plt.savefig('viz/written_spoken_digits/class_dist.png')
    plt.close()

    # Calculate audio data norm per coefficient
    # train_loader, test_loader = digits_get_data_loaders(200)
    # sum = 0
    # sum_squared = 0
    # n = 0
    # for i, (image_batch, audio_batch, label_batch) in enumerate(train_loader):
    #     if i ==0:
    #         print("Shape: ", audio_batch.sum((0,1,3)).shape)
    #     sum += audio_batch.sum((0,1,3))
    #     sum_squared += (audio_batch**2).sum((0,1,3))
    #     n += audio_batch.shape[0] * audio_batch.shape[1] * audio_batch.shape[3]
    # mean = sum / n
    # std = torch.sqrt((sum_squared / n) - (mean**2))
    # print("Mean: ", mean)
    # print("Std: ", std)
    
    sigmas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # Now create plot with subplots for weak_modality - image
    n_b = 4
    for sigma in sigmas:
        train_loader, test_loader = digits_get_data_loaders_weak_modality(4, sigma, weaken_image=True)
        image_batch, audio_batch, label_batch = next(iter(train_loader))
        fig, axes = plt.subplots(2, n_b, figsize=(10, 5))
        for i in range(n_b):
            axes[0, i].imshow(image_batch[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f"Label: {label_batch[i]}")
            axes[1, i].imshow(audio_batch[i].squeeze(), cmap='gray')
        if not os.path.exists('viz/written_spoken_digits_weak_image'):
            os.makedirs('viz/written_spoken_digits_weak_image')
        plt.savefig(f'viz/written_spoken_digits_weak_image/batch_sigma={sigma}.png')
        plt.close()

    # Now create plot with subplots for weak_modality - audio
    n_b = 4
    for sigma in sigmas:
        train_loader, test_loader = digits_get_data_loaders_weak_modality(4, sigma, weaken_audio=True)
        image_batch, audio_batch, label_batch = next(iter(train_loader))
        fig, axes = plt.subplots(2, n_b, figsize=(10, 5))
        for i in range(n_b):
            axes[0, i].imshow(image_batch[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f"Label: {label_batch[i]}")
            axes[1, i].imshow(audio_batch[i].squeeze(), cmap='gray')
        if not os.path.exists('viz/written_spoken_digits_weak_audio'):
            os.makedirs('viz/written_spoken_digits_weak_audio')
        plt.savefig(f'viz/written_spoken_digits_weak_audio/batch_sigma={sigma}.png')
        plt.close()

    # Now we plot data with noisy pairings
    ps = [0.01, 0.02, 0.04, 0.08, 0.10, 0.5]
    # Now create plot with subplots for weak_modality - image
    n_b = 4
    for p in ps:
        train_loader, test_loader = digits_get_data_loaders_noisy_pairing(4, p)
        image_batch, audio_batch, label_batch = next(iter(train_loader))
        fig, axes = plt.subplots(2, n_b, figsize=(10, 5))
        for i in range(n_b):
            axes[0, i].imshow(image_batch[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f"Label: {label_batch[i]}")
            axes[1, i].imshow(audio_batch[i].squeeze(), cmap='gray')
        if not os.path.exists('viz/written_spoken_digits_noisy_pairing'):
            os.makedirs('viz/written_spoken_digits_noisy_pairing')
        plt.savefig(f'viz/written_spoken_digits_noisy_pairing/batch_p={p}.png')
        plt.close()