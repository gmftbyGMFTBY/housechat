import torch
from models import *

def load_model(args):
    if args['model'] == 'bert':
        agent = BERTRetrievalAgent(args['total_steps'], args['multi_gpu'], run_mode=args['mode'], local_rank=args['local_rank'])
    else:
        pass
    return agent