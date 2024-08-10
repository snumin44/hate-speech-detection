import os
import re
import time
import random
import datetime
import argparse 
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append("../") 

from src.data_loader import (
        CustomDataset,
        DataCollator,
)  
from src.model import CRNN_Model
from utils.vocab import AddVocab
from utils.utils import (
        Jamo,
        get_score,
        test,
)

def argument_parser():

    # Required
    parser = argparse.ArgumentParser(description='hate speech detection')

    parser.add_argument('--test_path', type=str, default='../data/test_dataset.csv',
                        help='Path of test dataset'
                       )
    parser.add_argument('--model_path', type=str, default='../pretrained_model',
                        help='Directory for output'
                       )
    parser.add_argument('--vocab_path', type=str, default='../pretrained_model',
                        help='Directory for vocab'
                       )

    # Model settings
    parser.add_argument('--embed_dim', default=100, type=int,
                        help='Dimension of embedding'
                       )
    parser.add_argument('--num_kernels', default=100, type=int,
                        help='Number of kernels (CNN)'
                       )
    parser.add_argument('--kernel_sizes', nargs='+', default=[3, 4, 5], type=int,
                        help='Sizes of Kernels (CNN)'
                       )
    parser.add_argument('--stride', default=1, type=int,
                        help='Number of pixels by which the kernels shift over the input feature map (CNN)'
                       )    
    parser.add_argument('--gru_hidden_dim', default=100, type=int,
                        help='Dimesion of GRU (GRU)'
                       )    
    parser.add_argument('--gru_bidirectional', action="store_false", default=True,
                        help='Whether to use bidirectional GRU or not (GRU)'
                       )    

    # Train config    
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--shuffle', action="store_false", default=True, 
                        help='Batch size'
                       )
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Ratio of dropout'
                       )     
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default=42, type=int,
                        help = 'Random seed'
                       )  
    args = parser.parse_args()
    return args


def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    
    seed_everything(args)
       
    test_dataset = CustomDataset.load_csv_data(path=args.test_path, remove_blank=False)
    test_dataset.vocab = os.path.join(args.vocab_path, "vocab.json")
                
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=args.shuffle, collate_fn=DataCollator())
    
    num_labels = test_dataset.num_labels
    vocab_size = test_dataset.vocab_size
    
    if args.device == 'cuda':
        model = CRNN_Model(num_labels, vocab_size, args).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    
    else:
        model = CRNN_Model(num_labels, vocab_size, args)
        model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    
    scores = test(model, test_dataloader, args)
    print(f"Hate Speech Detection Accuracy : {round(scores['accuracy'] * 100, 2)} (%)")
    print(f"Hate Speech Detection Precision: {round(scores['precision'] * 100, 2)} (%)")
    print(f"Hate Speech Detection Recall   : {round(scores['recall'] * 100, 2)} (%)")


if __name__ == '__main__':
    args = argument_parser()
    main(args)