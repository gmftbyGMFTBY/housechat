from elasticsearch import Elasticsearch
from tqdm import tqdm
import ipdb, csv

class ESChat:
    
    '''basic elasticsearch searcher'''

    def __init__(self, index_name):
        self.es = Elasticsearch(http_auth=('elastic', 'elastic123'))
        self.index = index_name
        self.es.indices.put_settings(
            index=self.index,
            body={
                'index': {
                    'max_result_window': 500000,
                }
            }
        )
        
    def search(self, query, samples=10):
        '''
        query is the string, which contains the utterances of the conversation context.
        1. topic is a list contains the topic words
        2. query utterance msg
        
        context: query is Q-Q matching
        response: query is Q-A matching, which seems better
        '''
        query = query.replace('[SEP]', '')    # Need to replace the [SEP] berfore the searching
        dsl = {
            'query': {
                'match': {
                    'utterance': query    # Q-A matching is better
                }
            },
            # NOTE
            'collapse': {
                'field': 'keyword'
            }
        }
        begin_samples, rest = samples, []
        while len(rest) == 0:
            hits = self.es.search(index=self.index, body=dsl, size=begin_samples)['hits']['hits']
            if len(hits) == 0:
                ipdb.set_trace()
            for h in hits:
                item = {
                    'score': h['_score'], 
                    'utterance': h['_source']['utterance']
                }
                if item['utterance'] in query or 'http' in item['utterance']:
                    # avoid the repetive responses
                    continue
                else:
                    rest.append(item)
                # rest.append(item)
            begin_samples += 1
        return rest
    
def read_query(path):
    with open(path) as f:
        csv_f = csv.reader(f, delimiter='\t')
        dataset = [line[1] for line in csv_f if line]
    print(f'[!] find {len(dataset)} querys from {path}')
    return dataset

def read_post(path, query_datasets, mode='train'):
    with open(path) as f:
        csv_f = csv.reader(f, delimiter='\t')
        dataset, cache, cache_id, legal_counter = [], [], 0, 0
        for line in tqdm(csv_f):
            query = query_datasets[cache_id]
            if mode == 'train':
                session_id, _, utterance, label = line
                if int(session_id) == cache_id:
                    cache.append((utterance, int(label)))
                else:
                    labels = [i[1] for i in cache]
                    if 1 not in labels:
                        # eschat
                        rest = es.search(query)[0]
                        cache.append((rest, 1))
                        # ipdb.set_trace()
                    else:
                        legal_counter += 1
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
    print(f'[!] find {len(dataset)} responses from {path}; {round(legal_counter/len(dataset), 4)} samples are legal.')
    return dataset

def write_datasets(queries, responses):
    with open('train.query.tsv', 'w') as f:
        for idx, utterance in enumerate(queries):
            f.write(f'{idx}\t{utterance}\n')
    with open('train.reply.tsv', 'w') as f:
        for idx, session in enumerate(responses):
            for jdx, (utterance, label) in enumerate(session):
                f.write(f'{idx}\t{jdx}\t{utterance}\t{label}\n')
    
if __name__ == "__main__":
    es = ESChat('retrieval_database')
    queries = read_query('train/train.query.bak.tsv')
    responses = read_post('train/train.reply.bak.tsv', queries)
    write_datasets(queries, responses)
    
    