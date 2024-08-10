import os
import random
import asyncio
import argparse
import numpy as np
import torch

import sys
sys.path.append("../") 

from src.data_loader import CustomDataset
from src.model import CRNN_Model
from utils.utils import (
        Jamo,
        get_score,
        test,
)

def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_default_args():
    parser = argparse.ArgumentParser(description="hate speech detection")

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
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Ratio of dropout'
                       )     
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default=42, type=int,
                        help = 'Random seed'
                       )      
    
    args, _ = parser.parse_known_args()
    return args


async def inference(model, dataset, input_text, args):
    
    encoded_text = dataset.encode_text(input_text, encode=True, remove_blank=False)
    
    max_kernel_size = args.kernel_sizes[-1]
    if len(encoded_text) <= max_kernel_size:
        encoded_text = encoded_text + [dataset.vocab['<PAD>']] * 3
    encoded_text = torch.tensor(encoded_text).unsqueeze(0).to(args.device)
    
    model.eval()
    with torch.no_grad():
        logits = model(encoded_text)

    logits_index = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
    
    if logits_index[0] == 0: # clean text
        output = input_text
    else: output = '[알림] 해당 메시지는 규칙 위반으로 차단되었습니다.'
    
    return output


async def print_output(model, dataset, input_text, args):
    output = await inference(model, dataset, input_text, args)
    print(output)


async def main(args):

    seed_everything(args)
    
    # vocabulary
    dataset = CustomDataset.load_csv_data('../data/train_dataset_2.csv')
    dataset.vocab = os.path.join(args.vocab_path, "vocab.json")
    
    num_labels = dataset.num_labels
    vocab_size = dataset.vocab_size
    
    # model
    model = CRNN_Model(num_labels, vocab_size, args).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location=torch.device(args.device)))
    
    while True:
        input_text = input('Enter text to be inferenced (type "exit" to quit): ')
        
        if input_text.lower() == "exit":
            print("Exiting the inference loop.")
            break
        
        await print_output(model, dataset, input_text, args)
        

if __name__ == '__main__':
    args = get_default_args()
    asyncio.run(main(args))