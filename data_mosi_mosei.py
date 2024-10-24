import subprocess
import gdown
import os
import sys
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List
import torchvision.transforms as T
from matplotlib import pyplot as plt

def _padding(inputs: List):
    """
    Function to pad the input data to the same length. 
    We use this in the dataloader for the collate_fn argument.
    """
    processed_input = []
    processed_input_lengths = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        # pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(torch.stack(feature))

    for sample in inputs:
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input[0], processed_input[1], processed_input[2], torch.tensor(labels).view(len(inputs), 1)

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

def _drop_entry(dataset):
    """Drop entries where there's no text in the data and where the vision samples are corrupted."""
    drop = []
    n_start = dataset["text"].shape[0]
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    for ind, k in enumerate(dataset["vision"]):
        if np.any(np.abs(k)>10000):
            if ind not in drop:
                drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    dataset['audio'][dataset['audio'] == -np.inf] = 0.0
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    print(f"Dropped {n_start - dataset['text'].shape[0]} entries")
    return dataset

def _process_mosi(viz=True):
    """
    Cleans up the MOSI dataset and saves the cleaned data back to a .pkl file.
    If viz is set to True, it will also plot and save histograms of the values for each modality for each split and print out the mean and std for each modality.
    """
    # If the cleaned data already exists, load it and return
    if os.path.exists('data/mosi/mosi_raw_clean.pkl'):
        with open('data/mosi/mosi_raw_clean.pkl', 'rb') as f:
            return pickle.load(f)
        
    with open('data/mosi/mosi_raw.pkl', 'rb') as f:
        data = pickle.load(f)

    # Assuming data is a pandas DataFrame or can be converted to one
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data['train'] = _drop_entry(data['train'])
    data['valid'] = _drop_entry(data['valid'])
    data['test'] = _drop_entry(data['test'])

    if viz:
        # Plot and save histogram of values for each modality for train split
        if not os.path.exists("viz/mosi"):
            os.makedirs("viz/mosi")
        for modality in ['vision', 'audio', 'text']:
            for split in ['train', 'valid', 'test']:
                print(f"Plotting histogram for {modality} in {split} split")
                fig, ax = plt.subplots()
                num_bins = 100
                flat = data[split][modality].flatten()
                counts, bin_edges = np.histogram(flat, bins=num_bins)
                # Determine second largest count
                sorted_counts = np.sort(counts)
                second_largest = sorted_counts[-2]
                ax.hist(flat, bins=num_bins, alpha=0.7, color='blue')
                # ax.set_yscale('log')
                # ax.set_ylim(0, second_largest)
                plt.ylabel('Count')
                plt.xlabel('Value')
                plt.title(f'Histogram of {modality} values in {split} split')
                plt.savefig(f"viz/mosi/{split}_{modality}.png")
                ax.set_ylim(0, second_largest)
                plt.savefig(f"viz/mosi/{split}_{modality}_zoomed.png")
        # Now print out the mean and std for each modality
        for modality in ['vision', 'audio', 'text']:
            print(f"Train {modality} mean: {np.mean(data['train'][modality])}")
            print(f"Train {modality} std: {np.std(data['train'][modality])}")
    # Save the cleaned data back to a .pkl file
    with open('data/mosi/mosi_raw_clean.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data

def _process_mosei(viz=True):
    """
    Some of the data in the MOSEI dataset is corrupted. This function removes them and saves the cleaned data back to a .pkl file.
    """
    # If the cleaned data already exists, load it and return
    if os.path.exists('data/mosei/mosei_raw_clean.pkl'):
        with open('data/mosei/mosei_raw_clean.pkl', 'rb') as f:
            return pickle.load(f)
        
    with open('data/mosei/mosei_raw.pkl', 'rb') as f:
        data = pickle.load(f)

    # Assuming data is a pandas DataFrame or can be converted to one
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Drop entries where there's no text in the data and where the vision samples are corrupted
    # Also sets values in audio samples that are -inf to 0
    data['train'] = _drop_entry(data['train'])
    data['valid'] = _drop_entry(data['valid'])
    data['test'] = _drop_entry(data['test'])

    if viz:
        # Plot and save histogram of values for each modality for train split
        from matplotlib import pyplot as plt
        if not os.path.exists("viz/mosei"):
            os.makedirs("viz/mosei")
        for modality in ['vision', 'audio', 'text']:
            for split in ['train', 'valid', 'test']:
                print(f"Plotting histogram for {modality} in {split} split")
                fig, ax = plt.subplots()
                num_bins = 100
                flat = data[split][modality].flatten()
                counts, bin_edges = np.histogram(flat, bins=num_bins)
                # Determine second largest count
                sorted_counts = np.sort(counts)
                second_largest = sorted_counts[-2]
                ax.hist(flat, bins=num_bins, alpha=0.7, color='blue')
                # ax.set_yscale('log')
                # ax.set_ylim(0, second_largest)
                plt.ylabel('Count')
                plt.xlabel('Value')
                plt.title(f'Histogram of {modality} values in {split} split')
                plt.savefig(f"viz/mosei/{split}_{modality}.png")
                ax.set_ylim(0, second_largest)
                plt.savefig(f"viz/mosei/{split}_{modality}_zoomed.png")
        # Now print out the mean and std for each modality
        for modality in ['vision', 'audio', 'text']:
            print(f"Train {modality} mean: {np.mean(data['train'][modality])}")
            print(f"Train {modality} std: {np.std(data['train'][modality])}")
    # Save the cleaned data back to a .pkl file
    with open('data/mosei/mosei_raw_clean.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data

class MOSI_MOSEI(Dataset):
    """Implements Affect data as a torch dataset."""
    def __init__(self, data: Dict, aligned: bool = True, max_pad_num=50, transform: Dict = {'vision': None, 'audio': None, 'text': None}) -> None:
        """Instantiate AffectDataset

        Args:
            data (Dict): Data dictionary
            aligned (bool, optional): Whether to align data or not across modalities. Defaults to True.
            max_pad_num (int, optional): Maximum padding number. Defaults to 50.
            transform (Dict, optional): Dictionary of transformations. Defaults to {'vision': None, 'audio': None, 'text': None}.
        """
        self.dataset = data
        self.aligned = aligned
        self.max_pad_num = max_pad_num
        self.transform = transform

    def __getitem__(self, ind):
        """Get item from dataset."""
        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()
        
        tmp_label = self.dataset['labels'][ind]
        label = torch.tensor(tmp_label).float()

        if self.max_pad_num is not None:
            tmp = [vision, audio, text, label]
            for i in range(len(tmp) - 1):
                tmp[i] = tmp[i][:self.max_pad_num]
                tmp[i] = F.pad(tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0]))
        else:
            tmp = [vision, audio, text, ind, label]
        for i, key in enumerate(['vision', 'audio', 'text']):
            if self.transform[key] is not None:
                tmp[i] = self.transform[key](tmp[i])
        return tmp

    def __len__(self):
        """Get length of dataset."""
        return self.dataset['vision'].shape[0]

def get_mosi(batch_size=32, train_shuffle=True):
    _download_mosi()
    processed_dataset = _process_mosi()

    mosi_vision_norm = lambda x: (x - -0.260075) / 1.29176
    mosi_audio_norm = lambda x: (x - 0.710609) / 12.2957
    mosi_text_norm = lambda x: (x - -0.0006965) / 0.1557696
    transform = { # mean and std calculated in _process_mosi
        'vision': mosi_vision_norm,
        'audio': mosi_audio_norm,
        'text': mosi_text_norm,
        }
    
    mosi_train = MOSI_MOSEI(processed_dataset['train'], max_pad_num=50, aligned=True, transform=transform)
    train_loader = DataLoader(mosi_train, batch_size=batch_size, shuffle=train_shuffle, collate_fn=_padding)

    mosi_valid = MOSI_MOSEI(processed_dataset['valid'], max_pad_num=50, aligned=True, transform=transform)
    valid_loader = DataLoader(mosi_valid, batch_size=batch_size, shuffle=False, collate_fn=_padding)

    mosi_test = MOSI_MOSEI(processed_dataset['test'], max_pad_num=50, aligned=True, transform=transform)
    test_loader = DataLoader(mosi_test, batch_size=batch_size, shuffle=False, collate_fn=_padding)

    return train_loader, valid_loader, test_loader

def get_mosei(batch_size=32, train_shuffle=True):
    _download_mosei()
    processed_dataset = _process_mosei()

    mosei_vision_norm = lambda x: (x - 106.02776) / 256.87140
    mosei_audio_norm = lambda x: (x - 1.1677283) / 15.1068355
    mosei_text_norm = lambda x: (x - 0.0004857003) / 0.2076894670
    transform = { # mean and std calculated in _process_mosei
        'vision': mosei_vision_norm,
        'audio': mosei_audio_norm,
        'text': mosei_text_norm,
        }

    mosei_train = MOSI_MOSEI(processed_dataset['train'], max_pad_num=50, aligned=True, transform=transform)
    train_loader = DataLoader(mosei_train, batch_size=batch_size, shuffle=train_shuffle, collate_fn=_padding)

    mosei_valid = MOSI_MOSEI(processed_dataset['valid'], max_pad_num=50, aligned=True, transform=transform)
    valid_loader = DataLoader(mosei_valid, batch_size=batch_size, shuffle=False, collate_fn=_padding)

    mosei_test = MOSI_MOSEI(processed_dataset['test'], max_pad_num=50, aligned=True, transform=transform)
    test_loader = DataLoader(mosei_test, batch_size=batch_size, shuffle=False, collate_fn=_padding)

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

def viz_hist(mosi_or_mosei, data, type):
    # Now let's visualize the labels as a histogram, bins are from -3 to 3 (7 classes), with 20 bins
    fig, ax = plt.subplots()
    data = data.flatten()
    ax.hist(data, bins=25)
    ax.set_title(f"{type}")
    plt.savefig(f"viz/{mosi_or_mosei}/hist_{type}.png")


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
    viz_hist('mosi', train_loader.dataset.dataset['vision'], 'vision')
    viz_hist('mosi', train_loader.dataset.dataset['audio'], 'audio')
    viz_hist('mosi', train_loader.dataset.dataset['text'], 'text')
    viz_hist('mosi', train_loader.dataset.dataset['labels'], 'labels')

    # # Now for MOSEI
    train_loader, valid_loader, test_loader = mosei_get_data_loaders(n_batch) 
    # Let's visualize some of the training data
    batch = next(iter(train_loader))
    viz('mosei', batch)
    viz_hist('mosei', train_loader.dataset.dataset['vision'], 'vision')
    viz_hist('mosei', train_loader.dataset.dataset['audio'], 'audio')
    viz_hist('mosei', train_loader.dataset.dataset['text'], 'text')
    viz_hist('mosei', train_loader.dataset.dataset['labels'], 'labels')