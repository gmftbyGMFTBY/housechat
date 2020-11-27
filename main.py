from dataloader import *
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import argparse
from models import *
from utils import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()

def main(**args):
    if args['mode'] == 'train':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
        data = HouseChatDataset(mode='train', max_length=args['max_length'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, sampler=train_sampler, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
        args['total_steps'] = int(len(data) * args['epoch'] / args['batch_size'])
        
        agent = load_model(args)
        
        sum_writer = SummaryWriter(log_dir=f'rest/{args["model"]}')
        for i in tqdm(range(args['epoch'])):
            train_loss = agent.train_model(
                iter_, mode='train', recoder=sum_writer, idx_=i,
            )
            if args['local_rank'] == 0:
                agent.save_model(f'ckpt/{args["model"]}/best.pt')
        sum_writer.close()
    elif args['mode'] == 'test':
        args['total_steps'] = 0
        data = HouseChatDataset(mode='test', max_length=args['max_length'])
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
        agent = load_model(args)
        agent.load_model(f'ckpt/{args["model"]}/best.pt')
        rest_path = f'rest/{args["model"]}/rest.txt'
        test_loss = agent.test_model(iter_, rest_path)
    elif args['mode'] == 'generate':
        args['total_steps'] = 0
        data = HouseChatDataset(mode='test', max_length=args['max_length'])
        iter_ = DataLoader(data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate)
        agent = load_model(args)
        agent.load_model(f'ckpt/{args["model"]}/best.pt')
        rest_path = f'rest/{args["model"]}/generate.txt'
        test_loss = agent.generate(iter_, rest_path)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    main(**args)
