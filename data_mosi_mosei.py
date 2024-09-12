import subprocess
import gdown
import os
import sys
from torch.utils.data import DataLoader
import torch

def _get_MultiBench():
    if not os.path.exists('MultiBench'):
        # Clone Multi Bench
        subprocess.run("git clone git@github.com:pliang279/MultiBench.git", shell = True, executable="/bin/bash")
        print("Cloned MultiBench")
    else:
        print("MultiBench already downloaded")

def _download_mosi():
    if not os.path.exists('data/mosi'):
        os.makedirs('data/mosi')
        folder_url = "https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq"
        gdown.download_folder(url=folder_url, output='data/mosi', quiet=False, use_cookies=False)
    else:
        print("MOSI already downloaded")

def _download_mosei():
    if not os.path.exists('data/mosei'):
        os.makedirs('data/mosei')
        folder_url = "https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv"
        gdown.download_folder(url=folder_url, output='data/mosei', quiet=False, use_cookies=False)
    else:
        print("MOSEI already downloaded")

def get_mosi(batch_size=32, train_shuffle=True):
    _download_mosi()
    _get_MultiBench()
    sys.path.append(os.path.join(os.path.dirname(__file__), 'MultiBench')) # MultiBench doesnt have __init__.py so we need to add the path manually
    from MultiBench.datasets.affect.get_data import get_dataloader
    train_loader, valid_loader, test_loader = get_dataloader(
        'data/mosi/mosi_raw.pkl', 
        data_type='mosi', 
        max_pad=True, 
        max_seq_len=50,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
        z_norm=False,
        num_workers=1
        )
    return train_loader, valid_loader, test_loader

def get_mosei(batch_size=32, train_shuffle=True):
    _download_mosei()
    _get_MultiBench()
    sys.path.append(os.path.join(os.path.dirname(__file__), 'MultiBench')) # MultiBench doesnt have __init__.py so we need to add the path manually
    from MultiBench.datasets.affect.get_data import get_dataloader
    train_loader, valid_loader, test_loader = get_dataloader(
        'data/mosei/mosei_raw.pkl', 
        data_type='mosei', 
        max_pad=True, 
        max_seq_len=50,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
        z_norm=False,
        num_workers=1
        )
    return train_loader, valid_loader, test_loader

def viz(mosi_or_mosei, batch):
    # First lets check the shape of the data
    print(f"\n{mosi_or_mosei}:")
    print(f"Elements per batch: {len(batch)}")
    print(f"Shape of 0th element (Vision): {batch[0].shape}")
    print(f"Shape of 1st element (Audio): {batch[1].shape}")
    print(f"Shape of 2nd element (Text): {batch[2].shape}")
    print(f"Shape of 3rd element (Label): {batch[3].shape}")

    # Now let's visualize some of the training data
    # Note that these are features extracted from the raw data, so they might not make sense
    import matplotlib.pyplot as plt
    import numpy as np
    vision = batch[0].numpy()
    audio = batch[1].numpy()
    text = batch[2].numpy()
    label = batch[3].numpy()
    if not os.path.exists("viz/"+mosi_or_mosei):
        os.makedirs("viz/"+mosi_or_mosei)

    # First lets print the labels
    print(f"Labels: {label}")

    # Now lets visualize the vision features
    fig, axs = plt.subplots(n_batch, 1)
    fig.tight_layout()
    for i in range(n_batch):
        cax = axs[i].imshow(vision[i].T, aspect='auto')
        axs[i].set_title(f"Vision {i}")
        fig.colorbar(cax, ax=axs[i])
    plt.savefig(f"viz/{mosi_or_mosei}/vision.png")

    # Now let's visualize the audio features
    fig, axs = plt.subplots(n_batch, 1)
    fig.tight_layout()
    for i in range(n_batch):
        cax = axs[i].imshow(audio[i].T, aspect='auto')
        axs[i].set_title(f"Audio {i}")
        fig.colorbar(cax, ax=axs[i])
    plt.savefig(f"viz/{mosi_or_mosei}/audio.png")

    # Now let's visualize the text features
    fig, axs = plt.subplots(n_batch, 1)
    fig.tight_layout()
    for i in range(n_batch):
        cax = axs[i].imshow(text[i].T, aspect='auto')
        axs[i].set_title(f"Text {i}")
        fig.colorbar(cax, ax=axs[i])
    plt.savefig(f"viz/{mosi_or_mosei}/text.png")

    # Now let's visualize the labels as a histogram, bins are from -3 to 3 (7 classes)
    plt.hist(label, bins=np.arange(-3.5, 4.5, 1))
    plt.title("Labels")
    plt.savefig(f"viz/{mosi_or_mosei}/labels.png")


def mosi_get_data_loaders(batch_size):
    """
    Function to get the data loaders for the MOSI dataset.
    """
    train_loader, valid_loader, test_loader = get_mosi(batch_size=batch_size, train_shuffle=True)
    return train_loader, valid_loader, test_loader

def mosei_get_data_loaders(batch_size):
    """
    Function to get the data loaders for the MOSEI dataset.
    """
    train_loader, valid_loader, test_loader = get_mosei(batch_size=batch_size, train_shuffle=True)
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    n_batch = 4

    # Let's download the data and get the dataloaders
    # First for MOSI
    train_loader, valid_loader, test_loader = mosi_get_data_loaders(n_batch)
    # Let's visualize some of the training data
    batch = next(iter(train_loader))
    viz('mosi', batch)

    # # Now for MOSEI
    train_loader, valid_loader, test_loader = mosei_get_data_loaders(n_batch) 
    # Let's visualize some of the training data
    batch = next(iter(train_loader))
    viz('mosei', batch)