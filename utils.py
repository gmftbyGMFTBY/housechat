import torch
from models import *

def load_model(args):
    if args['model'] == 'bert':
        agent = BERTRetrievalAgent(args['total_steps'], run_mode=args['mode'], lr=args['lr'], grad_clip=args['gradc'])
    else:
        pass
    return agent
