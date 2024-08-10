import os
import sys
import json
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

sys.path.append("../")
from utils.utils import Jamo


class CustomDataset(Dataset):

    def __init__(self, texts, chars, label):
        self.texts = texts
        self.label = label
        self.chars = chars
        self._vocab = {}
                        
    @classmethod
    def load_csv_data(cls, path, sep='\t', remove_jongsung=True, remove_blank=True):
        #load data
        datasets = pd.read_csv(path, sep=sep)
        texts, label = list(datasets['text']), list(datasets['label'])

        # split text into jaeum and moeum.
        jamo = Jamo()
        chars = [jamo.split_jamo(text, remove_jongsung, remove_blank) for text in texts]
        return cls(texts, chars, label)
        
    def build_vocab(self, save_path):
        all_chars = [char for chars in self.chars for char in chars]
        vocab = Counter(all_chars)
        vocab = sorted(vocab, key=vocab.get, reverse=True)
        vocab = {word: idx + 3 for idx, word in enumerate(vocab)}
        vocab['<PAD>'], vocab['<UNK>'], vocab['<BOS>'] = 0, 1, 2
        self.vocab = vocab
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, "vocab.json")
        
        with open(save_path, 'w') as file:
            json.dump(vocab, file, indent=4)
        
    @property
    def vocab(self):
        return self._vocab

    @vocab.setter 
    def vocab(self, vocabulary):
        if isinstance(vocabulary, dict):
            self._vocab = vocabulary
        else:
            with open(vocabulary, 'r') as file:
                self._vocab= json.load(file)
        
    @property
    def vocab_size(self):
        return max(self.vocab.values()) + 1
    
    @property
    def num_labels(self):
        return len(list(set(self.label)))    
    
    def add_vocab_list(self, tokens: list):
        if self.vocab is None:
            raise NotImplemetedError('There is no existing vocabulary')
        else:
            for token in list(set(tokens)):
                if token in self.vocab: pass
                else:
                    self.vocab[token] = len(self.vocab)
                
    def add_vocab_dict(self, tokens: dict):
        if self.vocab is None:
            raise NotImplemetedError('There is no existing vocabulary')   
        else:
            for token in tokens.keys():
                if token in self.vocab:
                    for added_token in tokens[token]:
                        self.vocab[added_token] = self.vocab[token]
                else:
                    fixed_index = max(self.vocab.values()) + 1
                    for added_token in tokens[token]:
                        self.vocab[added_token] = fixed_index       

    def encode_chars(self, chars):
        chars = ['<BOS>'] + chars
        return [self.vocab.get(char, self.vocab['<UNK>']) for char in chars]
            
    def encode_text(self, text, encode=True, remove_jongsung=True, remove_blank=True):
        jamo = Jamo()
        chars = jamo.split_jamo(text, remove_jongsung, remove_blank)
        if encode:
            return self.encode_chars(chars)
        else:
            return ['<BOS>'] + chars
    
    def __len__(self):
        assert len(self.chars) == len(self.label)
        return len(self.chars)

    def __getitem__(self, index):
        text = self.encode_chars(self.chars[index])
        label = int(self.label[index])
        
        return {
            'text': text,
            'label': label
        }

class DataCollator(object):

    def __call__(self, samples):        
        text, label = [], []
        for sample in samples:
            label.append(sample['label'])
            text.append(torch.tensor(sample['text'], dtype=torch.long))
            
        padded_text = pad_sequence(text, batch_first=True)
        label = torch.tensor(label, dtype=torch.long)

        return {
            'text':padded_text,
            'label':label,
        }