from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os, csv, ipdb, torch
from transformers import BertTokenizer
from tqdm import tqdm

def read_query(path):
    with open(path) as f:
        csv_f = csv.reader(f, delimiter='\t')
        dataset = [line[1] for line in csv_f if line]
    print(f'[!] find {len(dataset)} querys from {path}')
    return dataset

def read_post(path, mode='train'):
    with open(path) as f:
        csv_f = csv.reader(f, delimiter='\t')
        dataset, cache, cache_id = [], [], 0
        for line in csv_f:
            if mode == 'train':
                session_id, _, utterance, label = line
                if int(session_id) == cache_id:
                    cache.append((utterance, int(label)))
                else:
                    dataset.append(cache)
                    cache = [(utterance, int(label))]
                    cache_id += 1
            else:
                session_id, _, utterance = line
                if int(session_id) == cache_id:
                    cache.append(utterance)
                else:
                    dataset.append(cache)
                    cache = [utterance]
                    cache_id += 1
        if cache:
            dataset.append(cache)
    print(f'[!] find {len(dataset)} responses from {path}')
    return dataset

class HouseChatDataset(Dataset):
    
    def __init__(self, mode='train', max_length=256):
        self.max_len, self.mode = max_length, mode
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.save_path = f'data/{mode}.pt'
        if os.path.exists(self.save_path):
            self.data = torch.load(self.save_path)
            print(f'[!] load preprocessed data from {self.save_path}')
            return
        # read
        querys = read_query(f'data/{mode}/{mode}.query.tsv')
        responses = read_post(f'data/{mode}/{mode}.reply.tsv', mode=mode)
        assert len(querys) == len(responses), f'[!] find inconsisent samples number'
        self.data = []
        
        if mode == 'train':
            for context, response in tqdm(list(zip(querys, responses))):
                items, labels = [(context, i[0]) for i in response], [i[1] for i in response]
                items = self.vocab.batch_encode_plus(items)
                ids, token_type_ids = items['input_ids'], items['token_type_ids']
                items = [self._length_limit(i, j) for i, j in zip(ids, token_type_ids)]
                for (ids, token_type_ids), label in zip(items, labels):
                    self.data.append({'ids': ids, 'label': label, 'token_type_ids': token_type_ids})
        else:
            for context, response in tqdm(list(zip(querys, responses))):
                items = [(context, i) for i in response]
                items = self.vocab.batch_encode_plus(items)
                ids, token_type_ids = items['input_ids'], items['token_type_ids']
                items = [self._length_limit(i, j) for i, j in zip(ids, token_type_ids)]
                self.data.append({'ids': ids, 'token_type_ids': token_type_ids})
                
        print(f'[!] load {len(self.data)} samples, save into {self.save_path}')
        torch.save(self.data, self.save_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
            
    def _length_limit(self, ids, token_type_ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
            token_type_ids = [token_type_ids[0]] + token_type_ids[-(self.max_len-1):]
        return (ids, token_type_ids)
    
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.mode == 'train':
            ids, token_type_ids, labels = [], [], []
            for i in batch:
                ids.append(torch.LongTensor(i['ids']))
                token_type_ids.append(torch.LongTensor(i['token_type_ids']))
                labels.append(i['label'])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            mask = self.generate_mask(ids)
            labels = torch.LongTensor(labels)
            if torch.cuda.is_available():
                ids, token_type_ids, mask, labels = ids.cuda(), token_type_ids.cuda(), mask.cuda(), labels.cuda()
            return ids, token_type_ids, mask, labels
        else:
            assert len(batch) == 1, '[!] test batch must be 1'
            batch = batch[0]
            ids, token_type_ids = batch['ids'], batch['token_type_ids']
            ids = pad_sequence([torch.LongTensor(i) for i in ids], batch_first=True, padding_value=self.vocab.pad_token_id)
            token_type_ids = pad_sequence([torch.LongTensor(i) for i in token_type_ids], batch_first=True, padding_value=self.vocab.pad_token_id)
            mask = self.generate_mask(ids)
            if torch.cuda.is_available():
                ids, token_type_ids, mask = ids.cuda(), token_type_ids.cuda(), mask.cuda()
            return ids, token_type_ids, mask
        
if __name__ == "__main__":
    data = HouseChatDataset()
    # train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    loader = DataLoader(data, shuffle=True, batch_size=4, collate_fn=data.collate)
    for batch in loader:
        ipdb.set_trace()
        
        