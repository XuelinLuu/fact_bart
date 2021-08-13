import sys
sys.path.append('.')

import json
from tqdm import tqdm
from datasets import load_dataset
from openie.StanfordOpenIE import StanfordOpenIE

def handel_save_data(openie, dataset, save_dir, file_type):

    with open('{}/{}.json'.format(save_dir, file_type), 'w') as fwriter:
        for i in tqdm(range(len(dataset))):
            id = dataset[i]['id']
            article = dataset[i]['article']
            highlights = dataset[i]['highlights']
            article_triples = openie.annotate(article)
            highlights_triples = openie.annotate(highlights)

            fwriter.write(
                json.dumps({
                    'id': id,
                    'article': article,
                    'highlights': highlights,
                    'article_triples': article_triples,
                    'highlights_triples': highlights_triples
                }) + '\n'
            )


if __name__ =='__main__':
    openie = StanfordOpenIE()
    datasets = load_dataset("cnn_dailymail", '3.0.0')
    # train_dataset, valid_dataset, test_dataset = datasets['train'], datasets['validation'], datasets['test']
    
    # handel_save_data(openie, train_dataset, 'data', 'train')
    # handel_save_data(openie, valid_dataset, 'data', 'valid')
    # handel_save_data(openie, test_dataset, 'data', 'test')
    
# train_dataset, valid_dataset, test_dataset = datasets['train']