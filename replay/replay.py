import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer


class CnnDLDataset(Dataset):
    def __init__(self, dataset, max_seq_len, tokenizer_path) -> None:
        self.max_seq_len = max_seq_len
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        self.dataset = self.handle_data(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def handle_data(self):
        new_dataset = []
        for i in range(len(self.dataset)):
            article = self.dataset[i]['article']
            highlights = self.dataset['highlights']
            
            article = self.text2tensor(article)
            input_ids, attention_mask = article['input_ids'], article['attention_mask']
            highlights = self.text2tensor(highlights)
            hl_input_ids = highlights['input_ids']
            new_dataset.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'hl_input_ids': hl_input_ids
            })


    def text2tensor(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > (self.max_seq_len - 2):
            tokens = tokens[:(self.max_seq_len - 2)]
        tokens = ['<s>'] + tokens + ['</s>']
        len_tokens = len(tokens)
        
        if len_tokens < self.max_seq_len:
            tokens += ['<pad>'] * (self.max_seq_len - len_tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        attention_mask = [1] * len_tokens + [0] * (self.max_seq_len - len_tokens)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int64)
        }




def get_rouge(model, tokenizer, dataset):
    for i in tqdm(range(len(dataset))):
        article = dataset[i]['article'].replace('\n', ' ')
        highlights = dataset[i]['highlights']

        tokens = tokenizer.tokenize(article)
        if len(tokens) > 1022:
            tokens = tokens[:1022]
        tokens = ['<s>'] + tokens + ['</s>']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = torch.tensor([input_ids], dtype=torch.int64)

        summary_ids = model.generate(input_ids, num_beams=4, length_penalty=2.0, no_repeat_ngram_size=3)
        text = tokenizer.decode(summary_ids.squeeze(0), skip_special_tokens=True)
        print(text)


model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


if __name__ == '__main__':
    datas = load_dataset('cnn_dailymail', '3.0.0')
    train_, valid_, test_ = datas['train'], datas['validation'], datas['test']

    train_dataset = CnnDLDataset(train_, 1024, 'facebook/bart-base')
    valid_dataset = CnnDLDataset(valid_, 1024, 'facebook/bart-base')
    test_dataset = CnnDLDataset(test_, 1024, 'facebook/bart-base')
    
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
