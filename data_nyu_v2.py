from torch.utils.data import DataLoader, Dataset, IterableDataset
import tensorflow_datasets as tfds
import torch
import os 
from torchvision import transforms
import numpy as np
from torchvision.datasets.utils import download_url
import matplotlib.pyplot as plt
from nyuv2 import NYUv2

rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
depth_transform = transforms.Compose([
            lambda x: np.expand_dims(x, axis=-1),
            lambda x: x.astype('float16'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[2.6738], std=[1.5146]),
        ])

class NyuDepthDataset(IterableDataset):
    """
    A PyTorch Dataset for the NYU Depth V2 dataset (https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
    We use the TensorFlow Datasets version of the dataset (https://www.tensorflow.org/datasets/catalog/nyu_depth_v2).
    When iterated over will return a tuple of the image and depth map as PyTorch tensors.
    If you want random shuffling please do this during loading of the dataset and not within the dataloader.
    Please set batch size above 50 for 'better' randomness, as shards have less than 50 samples each and the shards are shuffled but not records within each shard.
    """
    def __init__(self, tfds_data, has_label=False):
        data = tfds.as_numpy(tfds_data)
        self.data = data
        self.image_transform = rgb_transform
        self.depth_transform = depth_transform
        if has_label:
            self.get_sample = self.get_sample_with_label
        else:
            self.get_sample = self.get_sample_no_label

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.data_iter = iter(self.data)  # Create a new iterator
        return self
    
    def get_sample_no_label(self):
        sample = next(self.data_iter)
        # Convert to PyTorch tensors
        image, depth = sample['image'], sample['depth']
        image = self.image_transform(image)
        depth = self.depth_transform(depth)
        return image, depth
            
    def get_sample_with_label(self):
        sample = next(self.data_iter)
        # Convert to PyTorch tensors
        image, depth, label = sample['image'], sample['depth'], sample['label']
        image = self.image_transform(image)
        depth = self.depth_transform(depth)
        return image, depth, label

    def __next__(self):
        try:
            return self.get_sample()
        except StopIteration:
            # If the current file iterator is exhausted, move to the next file
            self.data_iter = iter(self.data)
            raise StopIteration

def download_nyu_v2():
    """
    Download the NYUv2 dataset if it is not already downloaded and return the train and test datasets.
    """
    train_set = tfds.load(name="nyu_depth_v2", split="train", as_supervised=False, data_dir='data', shuffle_files=True)
    test_set = tfds.load(name="nyu_depth_v2", split="validation", as_supervised=False, data_dir='data', shuffle_files=False)

    return train_set, test_set

def get_nyu_v2():
    """
    Get the train and test datasets.
    Note that we use a Resize transform only on the supervised set, 
    we will use RandomlyResizedCrop on the unsupervised set during training.
    """
    train_set, test_set = download_nyu_v2()
    ds_train = NyuDepthDataset(train_set)
    ds_test = NyuDepthDataset(test_set) # We dont actally use this.
    
    t = transforms.Compose([transforms.ToTensor(), transforms.Resize((240, 320))])
    ds_train_supervised = NYUv2('data/NYUv2',  download=True, train=True,
        rgb_transform=rgb_transform, seg_transform=t, depth_transform=depth_transform)
    ds_test_supervised = NYUv2('data/NYUv2', download=True, train=False,
        rgb_transform=rgb_transform, seg_transform=t, depth_transform=depth_transform)
    
    return ds_train, ds_test, ds_train_supervised, ds_test_supervised

def nyu_v2_get_data_loaders(batch_size: int):
    """
    Returns train and test set data loaders for the NYUv2 dataset (https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
    We use the TensorFlow Datasets version of the dataset (https://www.tensorflow.org/datasets/catalog/nyu_depth_v2) for unsupervised training.
    And https://github.com/xapharius/pytorch-nyuv2 for the supervised version of the dataset.
    The TensorFlow Datasets version is much larger (contains all of the rgb+depth images) but lacks the segmentation labels, 
    whereas the second version is much smaller but contains the segmentation labels.

    Args:
        batch_size: The batch size for the data loaders.

    Returns:
        train_loader: The data loader for the train set.
        test_loader: The data loader for the test set.
    """
    ds_train, ds_test, ds_train_supervised, ds_test_supervised = get_nyu_v2()

    # these dont need shuffling (handled through tdfs)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True)

    # these need shuffling
    train_supervised_loader = DataLoader(ds_train_supervised, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_supervised_loader = DataLoader(ds_test_supervised, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, train_supervised_loader, test_supervised_loader

def compute_norm():
    train_loader, test_loader, train_supervised_loader, test_supervised_loader = nyu_v2_get_data_loaders(128)
    mean = torch.zeros(3)  # Assuming 3 channels (RGB)
    std = torch.zeros(3)
    total_images = 0
    # Iterate through the DataLoader
    rgb_mean = 0.
    rgb_var = 0.
    depth_mean = 0.
    depth_var = 0.
    nb_samples = 0.
    batch_counter = 0
    for rgb, depth in train_loader:
        batch_samples = rgb.size(0)

        rgb_mean += rgb.mean((0,2,3))
        rgb_var += rgb.var((0,2,3))
        
        depth_mean += depth.mean((0,2,3))
        depth_var += depth.var((0,2,3))

        nb_samples += batch_samples
        batch_counter += 1

    rgb_mean /= batch_counter
    rgb_std = torch.sqrt(rgb_var / batch_counter)

    depth_mean /= batch_counter
    depth_std = torch.sqrt(depth_var / batch_counter)

    print(f'RGB mean: {rgb_mean}, RGB std: {rgb_std}')  
    print(f'Depth mean: {depth_mean}, Depth std: {depth_std}')


def viz(rgb_images, depth_images, labels=None):
    n_images = len(rgb_images)
    if labels is None:
        n_rows = 2  # 1 row for RGB, 1 row for depth
    else:
        n_rows = 3 # 1 row for RGB, 1 row for depth, 1 row for labels
    n_cols = min(4, n_images)  # Adjust columns if more images than columns

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    
    # Note: plotted images are normalized
    for i in range(n_images):
        # Normalize RGB images if they are in the range [1.0, 255.0]
        if rgb_images[i].max() > 1.0:
            rgb_image = rgb_images[i] / 255.0  # Normalize to [0, 1]
        else:
            rgb_image = rgb_images[i]  # Already in [0, 1] range
            
        # Plot RGB image
        ax_rgb = axes[0, i % n_cols]
        ax_rgb.imshow(rgb_image.permute(1, 2, 0))
        ax_rgb.axis('off')
        ax_rgb.set_title(f'RGB Image {i + 1}')

        # Plot Depth image
        ax_depth = axes[1, i % n_cols]
        ax_depth.imshow(depth_images[i].permute(1, 2, 0), cmap='gray')
        ax_depth.axis('off')
        ax_depth.set_title(f'Depth Image {i + 1}')

        # Plot segmentation masks (labels) if provided
        if labels is not None:
            ax_seg = axes[2, i % n_cols]
            ax_seg.imshow(labels[i].permute(1, 2, 0), cmap='gray')
            ax_seg.axis('off')
            ax_seg.set_title(f'Segmentation Mask {i + 1}')


        # Move to the next column
        if (i + 1) % n_cols == 0:
            # If we filled the columns, we need to break to the next row
            if (i + 1) < n_images:
                axes[0, i % n_cols].set_visible(False)
                axes[1, i % n_cols].set_visible(False)

    plt.tight_layout()
    if os.path.exists('viz/nyu_v2') is False:
        os.makedirs('viz/nyu_v2')
    if labels is None:
        plt.savefig('viz/nyu_v2/unsupervised.png')
    else:
        plt.savefig('viz/nyu_v2/supervised.png')

if __name__ == "__main__":
    train_loader, test_loader, train_supervised_loader, test_supervised_loader = nyu_v2_get_data_loaders(4)
    # Get the first batch of images and labels from the train loader and visualize them
    rgb_images, depth_images = next(iter(train_loader))
    viz(rgb_images, depth_images)

    rgb_images, depth_images, labels = next(iter(train_supervised_loader))
    viz(rgb_images, depth_images, labels=labels)