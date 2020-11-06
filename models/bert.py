from .header import *

'''Cross-Attention BertRetrieval'''

class BERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTRetrieval, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        logits = output[0]    # [batch, 2]
        return logits
    
class BERTRetrievalAgent:

    def __init__(self, total_steps, multi_gpu, run_mode='train', local_rank=0):
        super(BERTRetrievalAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 1e-5,
            'grad_clip': 1.0,
            'samples': 10,
            'multi_gpu': self.gpu_ids,
            'max_len': 256,
            'vocab_file': 'bert-base-chinese',
            'warmup_ratio': 0.1,
            'total_steps': total_steps,
            'pad': 0,
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTRetrieval(self.args['model'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = transformers.AdamW(
            self.model.parameters(), 
            lr=self.args['lr'],
        )
        self.criterion = nn.CrossEntropyLoss()
        if run_mode == 'train':
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer, 
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=int(self.args['warmup_ratio'] * total_steps), 
                num_training_steps=total_steps,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
            )
        pprint.pprint(self.args)
        
    def save_model(self, path):
        state_dict = self.model.module.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
        
    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, attn_mask, label = batch
            self.optimizer.zero_grad()
            output = self.model(cid, token_type_ids, attn_mask)    # [B, 2]
            loss = self.criterion(
                output, 
                label.view(-1),
            )
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)

    @torch.no_grad()
    def test_model(self, test_iter, path):
        self.model.eval()
        pbar = tqdm(test_iter)
        rest, batch_num = [], 0
        for idx, batch in enumerate(pbar):
            cid, token_type_ids, attn_mask = batch
            output = self.model(cid, token_type_ids, attn_mask)    # [batch, 2]
            output = (F.softmax(output, dim=-1)[:, 1] > 0.5).tolist()    # [batch]
            output = [1 if i else 0 for i in output]
            rest.extend(
                [f'{batch_num}\t{idx}\t{label}\n' for idx, label in zip(range(len(output)), output)]
            )
            batch_num += 1
        with open(path, 'w') as f:
            for line in rest:
                f.write(line)
        print(f'[!] write the rest into the file {path}')