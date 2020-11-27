from tqdm import tqdm

def read(path):
    with open(path) as f:
        datasets = [i.strip().split('\t') for i in f.readlines()]
    return datasets

def read_rest(path):
    with open(path) as f:
        rest = [i.strip().split('\t')[-1] for i in f.readlines()]
    return rest

def write_file(datas, labels, path):
    with open(path, 'w') as f:
        assert len(datas) == len(labels)
        for data, label in zip(datas, labels):
            data = '\t'.join(data)
            f.write(f'{data}\t{label}\n')
    print(f'[!] write {len(datas)} into {path}')

def read_dataset(path, path2):
    with open(path) as f:
        queries = [i.strip().split('\t') for i  in f.readlines()]
    with open(path2) as f:
        session_id, responses, cache = -1, [], []
        for line in f.readlines():
            line = line.strip().split('\t')
            if session_id == int(line[0]):
                if cache:
                    responses.append(cache)
                    cache = []
                session_id += 1
                cache.append(line[1:])    # no session id
            else:
        if cache:
            responses.append(cache)
    return queries, responses

if __name__ == "__main__":
    test_dataset = read('data/test/test.reply.tsv')
    test_label = read_rest('rest/bert/generate.txt')
    write_file(test_dataset, test_label, 'data/new_test.reply.tsv')

    # combine the test dataset and the train dataset
    train_dataset = read_dataset('data/train/train.query.tsv', 'data/train/train.reply.tsv')
    test_dataset = read_dataset('data/test/test/query.tsv', 'data/new_test.reply.tsv')

    # write into the data/train/
    with open('data/new_train.query.tsv', 'w') as f:
        session_id = 0
        for line in train_dataset[0] + test_dataset[0]:
            line = line[0]
            f.write(f'{session_id}\t{line}\n')
            session_id += 1

