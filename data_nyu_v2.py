import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.data import DataLoader, Dataset, IterableDataset
import tensorflow_datasets as tfds
import torch
import os 
from torchvision import transforms
import numpy as np
from torchvision.datasets.utils import download_url
import matplotlib.pyplot as plt
from nyuv2 import NYUv2

def set_tensorflow_not_use_gpu():
    import tensorflow as tf
    # Get the list of all available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Set no visible devices to prevent TensorFlow from using the GPU
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4876, 0.4176, 0.4005], std=[0.2895, 0.2978, 0.3118]),
            transforms.Resize((480, 640)),
        ])
depth_transform = transforms.Compose([
            lambda x: np.expand_dims(x, axis=-1) / 10.0, # Normalize depth values to [0, 1] and add channel dimension (like ToTensor does for RGB)
            transforms.ToTensor(),
            lambda x: x.to(torch.float32), # Convert to torch.float32
            # transforms.ConvertImageDtype(torch.float16),
            # lambda x: torch.where(torch.isinf(x), torch.tensor(10.0), x), # Convert 'inf' to 10 
            transforms.Normalize(mean=[0.2707], std=[0.1517]),
            transforms.Resize((480, 640)),
        ])

class NyuUnsupDataset(IterableDataset):
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
        # Set infinity to 10 in depth map
        depth = np.where(np.isinf(depth), 10.0, depth)
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
    set_tensorflow_not_use_gpu() # Prevent TensorFlow from using the GPU
    train_set = tfds.load(name="nyu_depth_v2", split="train", as_supervised=False, data_dir='data', shuffle_files=True)
    test_set = tfds.load(name="nyu_depth_v2", split="validation", as_supervised=False, data_dir='data', shuffle_files=False)

    return train_set, test_set

def get_nyu_v2(num_classes: int):
    """
    Get the train and test datasets.
    Note that we use a Resize transform only on the supervised set, 
    we will use RandomlyResizedCrop on the unsupervised set during training.
    """
    train_set, test_set = download_nyu_v2()
    ds_train = NyuUnsupDataset(train_set)
    ds_test = NyuUnsupDataset(test_set) # We dont actally use this.
    # To tensor, resize and one hot encode the segmentation labels
    t = transforms.Compose([transforms.ToTensor(), transforms.Resize((480, 640))])
    ds_train_supervised = NYUv2('data/NYUv2',  download=True, train=True, num_classes=num_classes,
        rgb_transform=rgb_transform, seg_transform=t, depth_transform=depth_transform)
    ds_test_supervised = NYUv2('data/NYUv2', download=True, train=False, num_classes=num_classes,
        rgb_transform=rgb_transform, seg_transform=t, depth_transform=depth_transform)
    
    return ds_train, ds_test, ds_train_supervised, ds_test_supervised

def nyu_v2_get_data_loaders(batch_size: int, num_classes: int, num_workers: int = 0):
    """
    Returns train and test set data loaders for the NYUv2 dataset (https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
    We use the TensorFlow Datasets version of the dataset (https://www.tensorflow.org/datasets/catalog/nyu_depth_v2) for unsupervised training.
    And https://github.com/xapharius/pytorch-nyuv2 for the supervised version of the dataset.
    The TensorFlow Datasets version is much larger (contains all of the rgb+depth images) but lacks the segmentation labels, 
    whereas the second version is much smaller but contains the segmentation labels.

    Args:
        batch_size: The batch size for the data loaders.
        num_classes: The number of classes in the dataset, either 40 or 13.

    Returns:
        train_loader: The data loader for the train set.
        test_loader: The data loader for the test set.
    """
    from tensorflow import compat
    compat.v1.logging.set_verbosity(compat.v1.logging.WARN) # Suppress TensorFlow warnings
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR) # Suppress TensorFlow warnings
    ds_train, ds_test, ds_train_supervised, ds_test_supervised = get_nyu_v2(num_classes)

    # these dont need shuffling (handled through tdfs)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # these need shuffling
    train_supervised_loader = DataLoader(ds_train_supervised, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_supervised_loader = DataLoader(ds_test_supervised, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
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


def viz(rgb_images, depth_images, num_classes, labels=None):
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
            ax_seg.imshow(labels[i], cmap='gray')
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
        plt.savefig(f'viz/nyu_v2/{num_classes}_classes_supervised.png')

if __name__ == "__main__":
    num_classes = 13
    train_loader, test_loader, train_supervised_loader, test_supervised_loader = nyu_v2_get_data_loaders(200, num_classes)
    # Get the first batch of images and labels from the train loader and visualize them
    rgb_images, depth_images = next(iter(train_loader))
    viz(rgb_images, depth_images, num_classes)

    rgb_images, depth_images, labels = next(iter(train_supervised_loader))
    # print max and min values for depth images
    viz(rgb_images, depth_images, num_classes, labels=labels)

    # num_classes = 40
    # train_loader, test_loader, train_supervised_loader, test_supervised_loader = nyu_v2_get_data_loaders(4, num_classes)
    # # Get the first batch of images and labels from the train loader and visualize them
    # rgb_images, depth_images, labels = next(iter(train_supervised_loader))    
    # viz(rgb_images, depth_images, num_classes, labels=labels)

    # Initialize histogram bins and counts
    rgb_bins = np.linspace(0.0, 1.0, 51)  # Assuming 8-bit RGB values
    depth_bins = np.linspace(0.0, 1.0, 51)  # Adjust range for depth values
    rgb_hist = np.zeros(50, dtype=np.int64)
    depth_hist = np.zeros(50, dtype=np.int64)

    # Process data in chunks
    i = 0
    n=0
    max_rbg = 0
    max_depth = 0
    min_rbg = 255
    min_depth = np.iinfo(np.uint16).max
    rgb_sum = np.array([0.0, 0.0, 0.0])
    depth_sum = 0.0
    rgb_sum_sq = np.array([0.0, 0.0, 0.0])
    depth_sum_sq = 0.0
    for rgb, depth, label in test_supervised_loader:
        i += 1
        n += rgb.shape[0] * rgb.shape[2] * rgb.shape[3]
        # Calculate mean and std
        rgb_sum += rgb.sum((0,2,3)).numpy()
        depth_sum += depth.sum((0,2,3)).numpy()
        rgb_sum_sq += (rgb**2).sum((0,2,3)).numpy()
        depth_sum_sq += (depth**2).sum((0,2,3)).numpy()
        
        # Process RGB data
        rgb_flat = rgb.numpy().flatten()
        rgb_chunk_hist, _ = np.histogram(rgb_flat, bins=rgb_bins)
        rgb_hist += rgb_chunk_hist
        
        # Process depth data
        depth_flat = depth.numpy().flatten()
        depth_chunk_hist, _ = np.histogram(depth_flat, bins=depth_bins)
        depth_hist += depth_chunk_hist

        if rgb_flat.max() > max_rbg:
            max_rbg = rgb_flat.max()
        if depth_flat.max() > max_depth:
            max_depth = depth_flat.max()
        if rgb_flat.min() < min_rbg:
            min_rbg = rgb_flat.min()
        if depth_flat.min() < min_depth:
            min_depth = depth_flat.min()

    print(f'Max RGB: {max_rbg}, Min RGB: {min_rbg}')
    print(f'Max Depth: {max_depth}, Min Depth: {min_depth}')
    rgb_mean = rgb_sum / n
    depth_mean = depth_sum / n
    
    rgb_sum_sq = rgb_sum_sq / n
    depth_sum_sq = depth_sum_sq / n
    print("rgb_sum_sq", rgb_sum_sq)
    print("depth_sum_sq", depth_sum_sq)
    rgb_std = np.sqrt(rgb_sum_sq - (rgb_mean**2))
    depth_std = np.sqrt(depth_sum_sq  - (depth_mean**2))
    print(f'RGB mean: {rgb_mean}, RGB std: {rgb_std}')
    print(f'Depth mean: {depth_mean}, Depth std: {depth_std}')

    # Plot histograms
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].bar(rgb_bins[:-1], rgb_hist, width=np.diff(rgb_bins), align='edge', color='blue', alpha=0.7)
    ax[0].set_title('RGB Image Pixel Values')
    ax[0].set_xlabel('Pixel Value')
    ax[0].set_ylabel('Frequency')

    ax[1].bar(depth_bins[:-1], depth_hist, width=np.diff(depth_bins), align='edge', color='red', alpha=0.7)
    ax[1].set_title('Depth Image Pixel Values')
    ax[1].set_xlabel('Pixel Value')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('viz/nyu_v2/histograms.png')