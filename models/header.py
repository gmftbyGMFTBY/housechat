import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter
import numpy as np
import math
import ipdb
from tqdm import tqdm
from copy import deepcopy
import pprint
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import Counter, OrderedDict
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal
import random
from transformers.modeling_gpt2 import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertModel, BertForMultipleChoice
from transformers import AutoModelWithLMHead, AutoTokenizer, OpenAIGPTLMHeadModel
import transformers
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model