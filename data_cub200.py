import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader
import torchfile
import numpy as np
import torch
import torchvision.transforms.functional as tvF

def load_t7_file(filepath):
    t7_data = torchfile.load(filepath)
    # Convert to PyTorch tensor if necessary
    if isinstance(t7_data, np.ndarray):
        return torch.from_numpy(t7_data)
    return t7_data

# def download_cub_200_2011(root, filename):
#     import gdown
#     url = "https://drive.google.com/uc?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE"
#     output = os.path.join(root, filename) 
#     if not os.path.exists(root):
#         os.makedirs(root)
#     if not os.path.exists(output):
#         gdown.download(url, output, quiet=False)

import requests
def download_cub_text(root, filename):
    owner = 'nicolalandro'
    repo = 'ntsnet-cub200'
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    response = requests.get(url)
    release_data = response.json()

    save_path = os.path.join(root, filename)
    for asset in release_data['assets']:
        asset_url = asset['browser_download_url']
        asset_name = asset['name']
        r = requests.get(asset_url, allow_redirects=True)
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print(f'Downloaded: {asset_name} to {save_path}')

import tensorflow_datasets as tfds
def download_cub():
    """
    Download the CUB-200-2011 dataset if it is not already downloaded and return the train and test datasets.
    """
    train_set = tfds.load(name="caltech_birds2011", split="train", as_supervised=False)
    test_set = tfds.load(name="caltech_birds2011", split="test", as_supervised=False)
    train_supervised_set = tfds.load(name="caltech_birds2011", split="train", as_supervised=True)
    test_supervised_set = tfds.load(name="caltech_birds2011", split="test", as_supervised=True)
    return train_set, test_set, train_supervised_set, test_supervised_set

class CustomNumpyDataset(Dataset):
    def __init__(self, numpy_arrays, transform=None):
        """
        Args:
            numpy_arrays (iterable): An iterable of NumPy arrays.
            transform (callable, optional): Optional transform to be applied to the samples.
        """
        self.numpy_arrays = numpy_arrays
        self.transform = transform

    def __len__(self):
        return len(self.numpy_arrays)

    def __getitem__(self, idx):
        # Get the sample
        sample = self.numpy_arrays[idx]
        
        # Convert the NumPy array to a PyTorch tensor
        sample_tensor = torch.from_numpy(sample)
        
        # Apply any transformations if provided
        if self.transform:
            sample_tensor = self.transform(sample_tensor)
        
        return sample_tensor
    
# class Cub_200_2011(Dataset):
#     def __init__(self, image_dir, text_dir, transform=None):
#         download_cub_200_2011('data', 'cub_200_2011')
#         self.image_dir = image_dir
#         self.text_dir = text_dir
#         self.transform = transform
        
#         self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.t7')])
#         self.text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.t7')])
        
#         self.class_to_image_files = self._group_files_by_class(self.image_files)
#         self.class_to_text_files = self._group_files_by_class(self.text_files)
        
#         self.data = self._create_data_list()

#     def _group_files_by_class(self, files):
#         class_to_files = {}
#         for file in files:
#             class_label = file[:3]
#             if class_label not in class_to_files:
#                 class_to_files[class_label] = []
#             class_to_files[class_label].append(file)
#         return class_to_files

#     def _create_data_list(self):
#         data = []
#         for class_label, image_files in self.class_to_image_files.items():
#             text_files = self.class_to_text_files.get(class_label, [])
#             num_images = len(image_files)
#             num_texts = len(text_files)
            
#             if num_texts == 0:
#                 raise ValueError(f"No text files found for class {class_label}")
            
#             for i in range(num_images):
#                 image_file = image_files[i]
#                 text_file = text_files[i % num_texts]
#                 data.append((image_file, text_file, class_label))
#         return data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_file, text_file, class_label = self.data[idx]
        
#         image_path = os.path.join(self.image_dir, image_file)
#         text_path = os.path.join(self.text_dir, text_file)
 
#         image = load_t7_file(image_path)
#         text = load_t7_file(text_path)
        
#         if self.transform:
#             image = self.transform(image)
#         print(text)
#         return image, text, class_label
def get_padding(image):
    max_w = 500
    max_h = 463
    
    imsize = image.size
    h_padding = (max_w - imsize[0]) / 2
    v_padding = (max_h - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    
    return padding

def cub_200_2011_get_data_loaders(batch_size: int):
    """
    Returns train and test set data loaders for the CUB-200-2011 dataset (https://data.caltech.edu/records/65de6-vp158)

    Args:
        batch_size: The batch size for the data loaders.

    Returns:
        train_loader: The data loader for the train set.
        test_loader: The data loader for the test set.
    """
    # download_cub_text('data', 'cub_200_2011_text')
    # train_set, test_set, train_supervised_set, test_supervised_set = download_cub()
    
    # ds_train = CustomNumpyDataset(tfds.as_numpy(train_set))
    # ds_test = CustomNumpyDataset(tfds.as_numpy(test_set))
    # ds_train_supervised = CustomNumpyDataset(tfds.as_numpy(train_supervised_set))
    # ds_test_supervised = CustomNumpyDataset(tfds.as_numpy(test_supervised_set))

    # train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    # train_supervised_loader = DataLoader(ds_train_supervised, batch_size=batch_size, shuffle=True)
    # test_supervised_loader = DataLoader(ds_test_supervised, batch_size=batch_size, shuffle=False)

    # return train_loader, test_loader, train_supervised_loader, test_supervised_loader
    from datasets import load_dataset
    ds = load_dataset("weiywang/CUB_200_2011_CAP", data_dir='data', cache_dir='data')
    split_ds = ds['train'].train_test_split(test_size=0.5) # Like the original paper
    ds_train = split_ds['train']
    ds_test = split_ds['test']

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def process(sample):
        image, text, label = sample['image'], sample['text'], sample['label']
        # Image from PIL to tensor
        image = tvF.pad(tvF.pil_to_tensor(image), get_padding(image))
        # Text from string to tensor
        print(type(image))
        print(type(label))
        return image, label
    
    ds_train.set_format(type='torch', columns=['image', 'text', 'label'])
    ds_test.set_format(type='torch', columns=['image', 'text', 'label'])
    
    ds_train.with_transform(process)
    ds_test.with_transform(process)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    
if __name__ == '__main__':
    train_loader, test_loader = cub_200_2011_get_data_loaders(32)
    # Produce a plot of the first batch of images with labels as their titles and annotations below them
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    # Get the first batch of images and labels from the train loader
    images, texts, class_labels = next(iter(train_loader))

    # Create a grid of images
    grid = make_grid(images, nrow=8)

    # Convert the grid to a numpy array and transpose it
    grid_np = grid.numpy().transpose((1, 2, 0))

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Display the grid of images
    ax.imshow(grid_np)

    # Set the title and annotations for each image
    for i in range(len(class_labels)):
        title = f"Label: {class_labels[i].item()}"
        text = texts[i].numpy().decode('utf-8')
        ax.text(i % 8, i // 8 + 1, title, fontsize=10, ha='center')
        ax.text(i % 8, i // 8 + 1.2, text, fontsize=8, ha='center')

    # Remove the axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Show the plot
    plt.savefig('viz/cub/cub_200_2011.png')