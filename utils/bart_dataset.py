import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FactDataset(Dataset):
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.datasets = self.read_csv()

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        data = self.datasets[idx]
        return (
            data['article_input_ids'],
            data['article_attention_mask'],
            data['highlights_input_ids'],
            data['highlights_attention_mask'],
            data['article_triples_input_ids'],
            data['article_triples_attention_mask'],
            data['article_triples_start_positions'],
            data['article_triples_start_positions_mask'],
            data['article_triples_label'],
            data['article_triples_label_mask']
        )

    def read_csv(self):
        datasets = []
        with open(self.data_path, 'r') as fread:
            for line in tqdm(fread.readlines(), desc='reading datasets...'):
                line = json.loads(line.strip())

                article_input_ids = torch.tensor(line['article_input_ids'], dtype=torch.int64)
                article_attention_mask = torch.tensor(line['article_attention_mask'], dtype=torch.int64)
                highlights_input_ids = torch.tensor(line['highlights_input_ids'], dtype=torch.int64)
                highlights_attention_mask = torch.tensor(line['highlights_attention_mask'], dtype=torch.int64)
                article_triples_input_ids = torch.tensor(line['article_triples_input_ids'], dtype=torch.int64)
                article_triples_attention_mask = torch.tensor(line['article_triples_attention_mask'], dtype=torch.int64)
                article_triples_start_positions = torch.tensor(line['article_triples_start_positions'], dtype=torch.int64)
                article_triples_start_positions_mask = torch.tensor(line['article_triples_start_positions_mask'], dtype=torch.int64)
                article_triples_label = torch.tensor(line['article_triples_label'], dtype=torch.int64)
                article_triples_label_mask = torch.tensor(line['article_triples_label_mask'], dtype=torch.int64)

                datasets.append({
                    'article_input_ids': article_input_ids,
                    'article_attention_mask': article_attention_mask,
                    'highlights_input_ids': highlights_input_ids,
                    'highlights_attention_mask': highlights_attention_mask,
                    'article_triples_input_ids': article_triples_input_ids,
                    'article_triples_attention_mask': article_triples_attention_mask,
                    'article_triples_start_positions': article_triples_start_positions,
                    'article_triples_start_positions_mask': article_triples_start_positions_mask,
                    'article_triples_label': article_triples_label,
                    'article_triples_label_mask': article_triples_label_mask
                })
        return datasets


if __name__ == '__main__':
    pass
    # tmp = FactDataset('data/tmp.json', 1024, 'facebook/bart-base')
    # print(tmp[0])
    # print(tmp[1])