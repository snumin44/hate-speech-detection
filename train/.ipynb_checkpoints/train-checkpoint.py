import os
import re
import time
import random
import logging
import datetime
import argparse 
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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

LOGGER = logging.getLogger()

def argument_parser():

    # Required
    parser = argparse.ArgumentParser(description='hate speech detection')

    parser.add_argument('--train_path', type=str, default='../data/train_dataset.csv',
                        help='Path of train dataset'
                       )
    parser.add_argument('--valid_path', type=str, default='../data/valid_dataset.csv',
                        help='Path of validation dataset'
                       )
    parser.add_argument('--test_path', type=str, default='../data/test_dataset.csv',
                        help='Path of test dataset'
                       )
    parser.add_argument('--output_path', type=str, default='../pretrained_model/sample',
                        help='Directory for output'
                       )
    parser.add_argument('--vocab_path', type=str, default='../pretrained_model/sample',
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
    parser.add_argument('--epochs', default=30, type=int,
                        help='Training epochs'
                       )   
    parser.add_argument('--early_stop', default=5, type=int,
                        help='Early stop'
                       )   
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--shuffle', action="store_false", default=True,
                        help='Load shuffled sequences'
                       )
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Ratio of dropout'
                       )   
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Weight decay'
                       )       
    parser.add_argument('--no_decay', nargs='+', default=['bias'],
                        help='List of parameters to exclude from weight decay' 
                       )              
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Leraning rate'
                       )       
    parser.add_argument('--eta_min', default=0, type=int,
                        help='Eta min for CosineAnnealingLR scheduler'
                       )   
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon for AdamW optimizer'
                       )   
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 42, type=int,
                        help = 'Random seed'
                       )  
    args = parser.parse_args()
    return args


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_adamw_optimizer(model, args):
    if args.no_decay: 
        # skip weight decay for some specific parameters i.e. 'bias'.
        no_decay = args.no_decay  
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        # weight decay for every parameter.
        optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.eps)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min, last_epoch=-1)
    return scheduler


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss


def train(model, train_dataloader, optimizer, scheduler, args):
    total_train_loss = 0
    
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
        # pass the data to device(cpu or gpu)            
        input_ids = batch['text'].to(args.device)
        label = batch['label'].to(args.device)

        optimizer.zero_grad()
        
        logits = model(input_ids)
                
        loss_fct = nn.CrossEntropyLoss()
        train_loss = loss_fct(logits, label.long())
        
        total_train_loss += train_loss.mean()
             
        train_loss.mean().backward()

        # Clip the norm of the gradients to 5.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  
        
        optimizer.step()    
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss


def valid(model, valid_dataloader, args):
    
    total_valid_loss = 0
    
    model.eval()    
    for _, batch in enumerate(valid_dataloader):
            
        input_ids = batch['text'].to(args.device)
        label = batch['label'].to(args.device)
            
        with torch.no_grad():
            logits = model(input_ids)

        loss_fct = nn.CrossEntropyLoss()
        valid_loss = loss_fct(logits, label.long())
            
        total_valid_loss += valid_loss.mean()
    
    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    return total_valid_loss    


def main(args):
    
    init_logging()
    seed_everything(args)
    
    LOGGER.info('*** Hate Speech Detection Model ***')    
    
    train_dataset = CustomDataset.load_csv_data(path=args.train_path)
    train_dataset.build_vocab(args.vocab_path)

    train_dataset.add_vocab_dict(AddVocab.CHOSUNG)
    train_dataset.add_vocab_dict(AddVocab.JUNGSUNG)
    train_dataset.add_vocab_dict(AddVocab.JONGSUNG)
    
    valid_dataset = CustomDataset.load_csv_data(path=args.valid_path)
    valid_dataset.vocab = train_dataset.vocab
    
    test_dataset = CustomDataset.load_csv_data(path=args.test_path)
    test_dataset.vocab = train_dataset.vocab
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=args.shuffle, collate_fn=DataCollator())
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=args.shuffle, collate_fn=DataCollator())
             
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=args.shuffle, collate_fn=DataCollator())
    
    num_labels = train_dataset.num_labels
    vocab_size = train_dataset.vocab_size
    
    if args.device == 'cuda':
        model = CRNN_Model(num_labels, vocab_size, args).to(args.device)
    else:
        model = CRNN_Model(num_labels, vocab_size, args)

    optimizer = get_adamw_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    early_stop_loss = list()
    best_loss, best_model = None, None
    
    t0 = time.time()

    for epoch_i in range(args.epochs):
        
        LOGGER.info(f'Epoch : {epoch_i+1}/{args.epochs}')

        train_loss = train(model, train_dataloader, optimizer, scheduler, args)
        valid_loss = valid(model, valid_dataloader, args)
        
        # Check Best Model
        if not best_loss or valid_loss < best_loss:
            best_loss = valid_loss
            best_model = deepcopy(model)
            
        # Early Stopping
        if len(early_stop_loss) == 0 or valid_loss > early_stop_loss[-1]:
            early_stop_loss.append(valid_loss)
            if len(early_stop_loss) == args.early_stop:break                                      
        else: early_stop_loss = list() 
                    
        print(f'Epoch:{epoch_i+1},Train_Loss:{round(float(train_loss.mean()), 4)},Valid_Loss:{round(float(valid_loss.mean()), 4)}') 
        
   # Save Best Model
    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
    torch.save(best_model.state_dict(), os.path.join(args.output_path, "pytorch_model.bin"))
    LOGGER.info(f'>>> Saved Best Model at {args.output_path}')
    
    training_time = format_time(time.time() - t0)
    print(f'Total Training Time:  {training_time}')
    
    scores = test(best_model, test_dataloader, args)
    print(f"Hate Speech Detection Accuracy : {round(scores['accuracy'] * 100, 2)} (%)")
    print(f"Hate Speech Detection Precision: {round(scores['precision'] * 100, 2)} (%)")
    print(f"Hate Speech Detection Recall   : {round(scores['recall'] * 100, 2)} (%)")


if __name__ == '__main__':
    LOGGER = logging.getLogger()
    args = argument_parser()
    main(args)