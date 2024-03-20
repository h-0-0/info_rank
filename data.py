import numpy as np
import torch
from torchvision import transforms
from typing import Tuple

def mnist_image_preprocess(images: np.ndarray) -> torch.Tensor:
    """
    Normalizes the MNIST images and turn into tensors.

    Args:
        - images (np.ndarray): MNIST images

    Returns:
        - images_normalized (troch.Tensor): The normalized MNIST images in tensor format

    """
    # We want to unflatten the images and normalize them
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.view(-1, 28, 28)),  # Unflatten the tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize pixel values
    ])
    print(images.shape)
    images = torch.tensor(images)
    print(images.shape)
    images_normalized = transform(images)
    return images_normalized

def fetch_data_partition(partition: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the (image, audio, label) data for the specified partition.

    Args:
        - partition (str): The partition to fetch the data for. Can be one of 'train' or 'test'

    Returns:
        - data (torch.Tensor): The data for the specified partition

    """
    if partition not in ['train', 'test']:
        raise ValueError("The partition should be one of 'train' or 'test'")
    # Load the data and return it
    images = np.load(f'data/data_wr_{partition}.npy')
    images = mnist_image_preprocess(images) # Normalize the images
    audio = torch.from_numpy(np.load(f'data/data_sp_{partition}.npy')) # Audio already pre-processed, just need to turn into tensor
    labels = torch.from_numpy(np.load(f'data/labels_{partition}.npy'))
    return images, audio, labels

def get_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the (image, audio, label) data for the training and test partitions.

    Data downlaoded from https://zenodo.org/records/3515935 and saved in the data folder.

    Returns:
        - data (torch.Tensor): The data for the training and test partitions

    """
    # Load the data and return it
    images_train, audio_train, labels_train = fetch_data_partition('train')
    images_test, audio_test, labels_test = fetch_data_partition('test')
    return images_train, audio_train, labels_train, images_test, audio_test, labels_test

if __name__ == '__main__':
    """
    Following code is for testing the data fetching and pre-processing functions, simply run this file to test them.
    Produces some visualizations of the data and saves some of the audio in listenable format.
    """
    # Load the data
    images_train, audio_train, labels_train, images_test, audio_test, labels_test = get_data()
    # Print the shapes of the data
    print("Training data shapes, images:", images_train.shape, ", audio:", audio_train.shape, ", labels:", labels_train.shape)
    print("Test data shapes, images:", images_test.shape, ", audio:", audio_test.shape, ", labels:", labels_test.shape)  
    # Plot a sample of the images and audio with their labels
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axs[0, i].imshow(images_train[i])
        axs[0, i].set_title(f"Label: {labels_train[i]}")
        axs[1, i].plot(audio_train[i])
    import os
    if not os.path.exists('viz'):
        os.makedirs('viz')
    plt.savefig('viz/data_sample.png')
    # Save some of the audio in listenable format, is pre-processed so it will sound like a beep
    from scipy.io.wavfile import write
    for i in range(5):
        write(f'viz/audio_sample_{i}.wav', 44100, audio_train[i].numpy())  # Assuming a sample rate of 44100 Hz