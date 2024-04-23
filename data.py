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
        transforms.Lambda(lambda x: x.view(-1, 1, 28, 28)),  # Unflatten the tensor
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize pixel values
    ])
    images = torch.tensor(images, dtype=torch.float32)
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
    audio = torch.tensor(np.load(f'data/data_sp_{partition}.npy'), dtype=torch.float32).reshape(-1, 1, 39, 13) # Audio already pre-processed, just need to turn into tensor
    labels = torch.tensor(np.load(f'data/labels_{partition}.npy'), dtype=torch.long)
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

def get_data_loaders(batch_size:int=32) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns the data loaders for the training and test partitions.

    Returns:
        - train_loader (torch.utils.data.DataLoader): The data loader for the training partition
        - test_loader (torch.utils.data.DataLoader): The data loader for the test partition

    """
    # Load the data
    images_train, audio_train, labels_train, images_test, audio_test, labels_test = get_data()
    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(list(zip(images_train, audio_train, labels_train)), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(list(zip(images_test, audio_test, labels_test)), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

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
        axs[0, i].imshow(images_train[i][0])
        axs[0, i].set_title(f"Label: {labels_train[i]}")
        axs[1, i].imshow(audio_train[i][0])
    import os
    if not os.path.exists('viz'):
        os.makedirs('viz')
    plt.savefig('viz/data_sample.png')

    # Now augment some of the data and plot it
    from loss import aug
    image1, audio1, image2, audio2 = aug(images_train, audio_train)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axs[0, i].imshow(image1[i][0])
        axs[1, i].imshow(audio1[i][0])
    plt.savefig('viz/data_augmented_sample.png')