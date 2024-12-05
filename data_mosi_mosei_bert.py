import sys
import os
import re
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call, CalledProcessError

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']


# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK


def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()





class MOSI:
    def __init__(self, dataset_dir='data', sdk_dir='data/CMU-MultimodalSDK', word_emb_path='data/glove.840B.300d.txt'):

        if sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(sdk_dir))
        print(sys.path)
        DATA_PATH = str(dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)
    
        except:
            # git clone sdk repo 
            if not os.path.exists(sdk_dir):
                check_call(' '.join(['git', 'clone', 'https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git', sdk_dir]), shell=True)
            import mmsdk
            from mmsdk import mmdatasdk as md
            if not os.path.exists(word_emb_path):
                check_call(' '.join(['wget', 'http://nlp.stanford.edu/data/glove.840B.300d.zip']), shell=True)
                check_call(' '.join(['unzip', 'glove.840B.300d.zip']), shell=True)
                check_call(' '.join(['mv', 'glove.840B.300d.txt', word_emb_path]), shell=True)
            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)


            # download highlevel features, low-level (raw) data and labels for the dataset MOSI
            # if the files are already present, instead of downloading it you just load it yourself.
            # here we use CMU_MOSI dataset as example.
            DATASET = md.cmu_mosi
            try:
                md.mmdataset(DATASET.highlevel, DATA_PATH)
            except RuntimeError:
                print("High-level features have been downloaded previously.")

            try:
                md.mmdataset(DATASET.raw, DATA_PATH)
            except RuntimeError:
                print("Raw data have been downloaded previously.")
                
            try:
                md.mmdataset(DATASET.labels, DATA_PATH)
            except RuntimeError:
                print("Labels have been downloaded previously.")
            
            # define your different modalities - refer to the filenames of the CSD files
            visual_field = 'CMU_MOSI_Visual_Facet_41'
            acoustic_field = 'CMU_MOSI_COVAREP'
            text_field = 'CMU_MOSI_TimestampedWords'


            features = [
                text_field, 
                visual_field, 
                acoustic_field
            ]

            recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
            print(recipe)
            dataset = md.mmdataset(recipe)

            # we define a simple averaging function that does not depend on intervals
            def avg(intervals: np.array, features: np.array) -> np.array:
                try:
                    return np.average(features, axis=0)
                except:
                    return features

            # first we align to words with averaging, collapse_function receives a list of functions
            dataset.align(text_field, collapse_functions=[avg])

            label_field = 'CMU_MOSI_Opinion_Labels'

            # we add and align to lables to obtain labeled segments
            # this time we don't apply collapse functions so that the temporal sequences are preserved
            label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
            dataset.add_computational_sequences(label_recipe, destination=None)
            dataset.align(label_field)

            # obtain the train/dev/test splits - these splits are based on video IDs
            train_split = DATASET.standard_folds.standard_train_fold
            dev_split = DATASET.standard_folds.standard_valid_fold
            test_split = DATASET.standard_folds.standard_test_fold


            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)\[.*\]')
            num_drop = 0 # a counter to count how many data points went into some processing issues

            for segment in dataset[label_field].keys():
                
                # get the video ID and the features out of the aligned dataset
                vid = re.search(pattern, segment).group(1)
                label = dataset[label_field][segment]['features']
                _words = dataset[text_field][segment]['features']
                _visual = dataset[visual_field][segment]['features']
                _acoustic = dataset[acoustic_field][segment]['features']

                # if the sequences are not same length after alignment, there must be some problem with some modalities
                # we should drop it or inspect the data again
                if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
                    print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
                    num_drop += 1
                    continue

                # remove nan values
                label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word in enumerate(_words):
                    if word[0] != b'sp':
                        actual_words.append(word[0].decode('utf-8'))
                        words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
                        visual.append(_visual[i, :])
                        acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)


                # z-normalization per instance and remove nan/infs
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if vid in train_split:
                    train.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in test_split:
                    test.append(((words, visual, acoustic, actual_words), label, segment))
                else:
                    print(f"Found video that doesn't belong to any splits: {vid}")

            print(f"Total number of {num_drop} datapoints have been dropped.")

            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()




class MOSEI:
    def __init__(self, dataset_dir='data', sdk_dir='data/CMU-MultimodalSDK', word_emb_path='data/glove.840B.300d.txt'):

        if sdk_dir is None:
            print("SDK path is not specified! Please specify first in constants/paths.py")
            exit(0)
        else:
            sys.path.append(str(sdk_dir))
        DATA_PATH = str(dataset_dir)
        CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/train.pkl')
            self.dev = load_pickle(DATA_PATH + '/dev.pkl')
            self.test = load_pickle(DATA_PATH + '/test.pkl')
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)

        except:
            # git clone sdk repo 
            if not os.path.exists(sdk_dir):
                check_call(' '.join(['git', 'clone', 'https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git', sdk_dir]), shell=True)
            import mmsdk
            from mmsdk import mmdatasdk as md

            # create folders for storing the data
            if not os.path.exists(DATA_PATH):
                check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)


            # download highlevel features, low-level (raw) data and labels for the dataset MOSEI
            # if the files are already present, instead of downloading it you just load it yourself.
            DATASET = md.cmu_mosei
            try:
                md.mmdataset(DATASET.highlevel, DATA_PATH)
            except RuntimeError:
                print("High-level features have been downloaded previously.")

            try:
                md.mmdataset(DATASET.raw, DATA_PATH)
            except RuntimeError:
                print("Raw data have been downloaded previously.")
                
            try:
                md.mmdataset(DATASET.labels, DATA_PATH)
            except RuntimeError:
                print("Labels have been downloaded previously.")
            
            # define your different modalities - refer to the filenames of the CSD files
            visual_field = 'CMU_MOSEI_VisualFacet42'
            acoustic_field = 'CMU_MOSEI_COVAREP'
            text_field = 'CMU_MOSEI_TimestampedWords'


            features = [
                text_field, 
                visual_field, 
                acoustic_field
            ]

            recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
            print(recipe)
            dataset = md.mmdataset(recipe)

            # we define a simple averaging function that does not depend on intervals
            def avg(intervals: np.array, features: np.array) -> np.array:
                try:
                    return np.average(features, axis=0)
                except:
                    return features

            # first we align to words with averaging, collapse_function receives a list of functions
            dataset.align(text_field, collapse_functions=[avg])

            label_field = 'CMU_MOSEI_LabelsSentiment'

            # we add and align to lables to obtain labeled segments
            # this time we don't apply collapse functions so that the temporal sequences are preserved
            label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
            dataset.add_computational_sequences(label_recipe, destination=None)
            dataset.align(label_field)

            # obtain the train/dev/test splits - these splits are based on video IDs
            train_split = DATASET.standard_folds.standard_train_fold
            dev_split = DATASET.standard_folds.standard_valid_fold
            test_split = DATASET.standard_folds.standard_test_fold


            # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
            EPS = 1e-6

            

            # place holders for the final train/dev/test dataset
            self.train = train = []
            self.dev = dev = []
            self.test = test = []
            self.word2id = word2id

            # define a regular expression to extract the video ID out of the keys
            pattern = re.compile('(.*)\[.*\]')
            num_drop = 0 # a counter to count how many data points went into some processing issues

            for segment in dataset[label_field].keys():
                
                # get the video ID and the features out of the aligned dataset
                try:
                    vid = re.search(pattern, segment).group(1)
                    label = dataset[label_field][segment]['features']
                    _words = dataset[text_field][segment]['features']
                    _visual = dataset[visual_field][segment]['features']
                    _acoustic = dataset[acoustic_field][segment]['features']
                except:
                    continue

                # if the sequences are not same length after alignment, there must be some problem with some modalities
                # we should drop it or inspect the data again
                if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
                    print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
                    num_drop += 1
                    continue

                # remove nan values
                label = np.nan_to_num(label)
                _visual = np.nan_to_num(_visual)
                _acoustic = np.nan_to_num(_acoustic)

                # remove speech pause tokens - this is in general helpful
                # we should remove speech pauses and corresponding visual/acoustic features together
                # otherwise modalities would no longer be aligned
                actual_words = []
                words = []
                visual = []
                acoustic = []
                for i, word in enumerate(_words):
                    if word[0] != b'sp':
                        actual_words.append(word[0].decode('utf-8'))
                        words.append(word2id[word[0].decode('utf-8')]) # SDK stores strings as bytes, decode into strings here
                        visual.append(_visual[i, :])
                        acoustic.append(_acoustic[i, :])

                words = np.asarray(words)
                visual = np.asarray(visual)
                acoustic = np.asarray(acoustic)

                # z-normalization per instance and remove nan/infs
                visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
                acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

                if vid in train_split:
                    train.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in dev_split:
                    dev.append(((words, visual, acoustic, actual_words), label, segment))
                elif vid in test_split:
                    test.append(((words, visual, acoustic, actual_words), label, segment))
                else:
                    print(f"Found video that doesn't belong to any splits: {vid}")
                

            print(f"Total number of {num_drop} datapoints have been dropped.")

            word2id.default_factory = return_unk

            # Save glove embeddings cache too
            self.pretrained_emb = pretrained_emb = load_emb(word2id, word_emb_path)
            torch.save((pretrained_emb, word2id), CACHE_PATH)

            # Save pickles
            to_pickle(train, DATA_PATH + '/train.pkl')
            to_pickle(dev, DATA_PATH + '/dev.pkl')
            to_pickle(test, DATA_PATH + '/test.pkl')

    def get_data(self, mode):

        if mode == "train":
            return self.train, self.word2id, self.pretrained_emb
        elif mode == "dev":
            return self.dev, self.word2id, self.pretrained_emb
        elif mode == "test":
            return self.test, self.word2id, self.pretrained_emb
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class MSADataset(Dataset):
    def __init__(self, dataset, split):

        ## Fetch dataset
        if dataset == "mosi":
            dataset = MOSI()
        elif dataset == "mosei":
            dataset = MOSEI()
        else:
            print("Dataset not defined correctly")
            exit()
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(split)
        self.len = len(self.data)

        self.visual_size = self.data[0][0][1].shape[1]
        self.acoustic_size = self.data[0][0][2].shape[1]

        self.word2id = self.word2id
        self.pretrained_emb = self.pretrained_emb


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def get_loader(dataset, split, batch_size, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(dataset, split)
    
    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things

        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        # sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])

        ## BERT-based features input prep

        # SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer
        sentences = [sample[0][3] for sample in batch]

        # bert_details = []
        # for sample in batch:
        #     text = " ".join(sample[0][3])
        #     encoded_bert_sent = bert_tokenizer.encode_plus(
        #         text, max_length=SENT_LEN+2, add_special_tokens=True, padding='max_length')
        #     bert_details.append(encoded_bert_sent)


        # Bert things are batch_first
        # bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        # bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        # bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])


        # lengths are useful later in using RNNs
        # lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        # Change the order of the dimensions in the visual and acoustic data so its batch first
        # sentences = sentences.permute(1, 0)
        visual = visual.permute(1, 0, 2)
        acoustic = acoustic.permute(1, 0, 2)

        return visual, acoustic, sentences, labels #, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader

def mosi_get_data_loaders_bert(batch_size):
    """
    Function to get the data loaders for the MOSI dataset.
    """
    train_loader = get_loader('mosi', 'train', batch_size, shuffle=True)
    dev_loader = get_loader('mosi', 'dev', batch_size, shuffle=False)
    test_loader = get_loader('mosi', 'test', batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader

def mosei_get_data_loaders_bert(batch_size):
    """
    Function to get the data loaders for the MOSEI dataset.
    """
    train_loader = get_loader('mosei', 'train', batch_size, shuffle=True)
    dev_loader = get_loader('mosei', 'dev', batch_size, shuffle=False)
    test_loader = get_loader('mosei', 'test', batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = mosi_get_data_loaders_bert(16)
    batch = next(iter(train_loader))
    print(len(batch))
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)
    print(batch[3].shape)
    print(batch[4].shape)
    print(batch[5].shape)
    print(batch[6].shape)
    print(batch[7].shape)
    # Find number of samples in train_loader
    num_samples = 0
    for i in train_loader:
        num_samples += i[0].shape[0]
    print(num_samples)