import json
from rouge import Rouge
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class FactDataset(Dataset):
    '''
    1. article triples 去重：
    2. 使用 rouge 获取 triples mask，某个排名阈值并且 rouge 分数小于阈值的为0，其他为 1
    3. 拼接 article triples
    4. 获取 article，highlights，article triples的词向量，外加article triples中 <s> 的位置
    '''
    def __init__(self, data_path, max_seq_len, tokenizer_path, num_triples_per_document=16) -> None:
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.num_triples_per_document = num_triples_per_document
        self.rouge = Rouge()
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
                line = json.loads(line)
                article = line['article']
                article_triples = line['article_triples']
                highlights = line['highlights']
                highlights_triples = line['highlights_triples']

                # article_triples 去重，type = list
                article_triples = self.remove_repeat_triple(article_triples)

                # 获取 article triple 在 highlights 中的 rouge
                # type(article_triples) = str
                # type(article_triples_label) = torch.Tensor(list)
                # type(article_triples_label_mask) = torch.Tensor(list)
                article_triples, article_triples_label, article_triples_label_mask = self.get_top_k_triple(article_triples, highlights, k_num=5)

                article_input_ids, article_attention_mask = self.text2tensor(article)
                highlights_input_ids, highlights_attention_mask = self.text2tensor(highlights)

                article_triples_input_ids, article_triples_attention_mask, article_triples_start_positions, article_triples_start_positions_mask = self.triples2tensor(article_triples)

                # TODO: article_triple_label 中 CLS 个数大于 article_triples_s_positions 中 CLS 的情况
                # 要根据 article_triples_s_positions 的数量标记 article_triples_start 值的有效位
                # triple_hidden_states + article_triple_s_positions + article_triples_attention_mask
                # => document 中所有三元组的CLS
                assert len(article_triples_start_positions) == self.num_triples_per_document, 'triples_start_positions != num_triples'
                assert len(article_triples_start_positions_mask) == self.num_triples_per_document, 'triples_start_positions_mask != num_triples'
                assert len(article_triples_label) == self.num_triples_per_document, 'triples_label != num_triples'
                assert len(article_triples_label_mask) == self.num_triples_per_document, 'triples_label_mask != num_triples'

                datasets.append({
                    'article_input_ids': article_input_ids,
                    'article_attention_mask': article_attention_mask,
                    'highlights_input_ids': highlights_input_ids,
                    'highlights_attention_mask': highlights_attention_mask,
                    'article_triples_input_ids': article_triples_input_ids,
                    'article_triples_attention_mask': article_triples_attention_mask,
                    'article_triples_start_positions': article_triples_start_positions,  # triples start 的位置
                    'article_triples_start_positions_mask': article_triples_start_positions_mask,  # 有效 start 位置个数
                    'article_triples_label': article_triples_label,  # 有效 start 位置标签
                    'article_triples_label_mask': article_triples_label_mask  # start 位置有效标签个数 （暂时无用，与positions_mask同）
                })
        return datasets

    def get_top_k_triple(self, article_triples, highlighs, k_num=5, rule='l', threshold=1):
        # threshold 默认值为1
        ts = []
        triple_tgt_label = [1] * self.num_triples_per_document
        for triple in article_triples:
            subj = triple['subject']
            rel = triple['relation']
            obj = triple['object']
            triple = '{} {} {}'.format(subj, rel, obj)
            scores = self.rouge.get_scores(triple, highlighs)
            rouge1, rouge2, rougel = scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']
            ts.append([triple, rouge1, rouge2, rougel])

        if rule == '1' or rule == 'rouge-1':
            ts.sort(key=lambda x: -x[1])
        elif rule == '2' or rule == 'rouge-2':
            ts.sort(key=lambda x: -x[2])
        elif rule == 'l' or rule == 'rouge-l':
            ts.sort(key=lambda x: -x[3])
        else:
            raise ValueError('sort rule must in (1, 2, l, rouge-1, rouge-2, rouge-l)')
        # TODO document 中三元组的个数小于 num_triples_per_document
        ts = ts[:self.num_triples_per_document]  # 每一个 document 中三元组的个数

        for i in range(k_num, len(article_triples)):
            score = ts[i][1] if (rule == '1' or rule == 'rouge-1') else (ts[i][2] if (rule == '2' or rule == 'rouge-2') else ts[i][3])
            if score < threshold:
                triple_tgt_label[i] = 0

        # assert len(ts) == len(triple_tgt_label), 'length of triples must equal to length of mask'
        # triple_tgt_label_mask 表示 document 中三元组的个数
        triple_tgt_label_mask = [1] * len(ts) + [0] * (self.num_triples_per_document - len(ts))
        sorted_ts = [t[0] for t in ts]
        return (
            '<s>'.join(sorted_ts),
            torch.tensor(triple_tgt_label, dtype=torch.int64),
            torch.tensor(triple_tgt_label_mask, dtype=torch.int64)
        )

    def remove_repeat_triple(self, triples):
        # remove repeat triple from article
        repeat_mask = [0] * len(triples)
        for i in range(len(triples)):
            for j in range(len(triples)):
                if i == j or repeat_mask[i] == 1:
                    continue
                if triples[i]['subjectSpan'][0] == triples[j]['subjectSpan'][0] \
                        and triples[i]['subjectSpan'][1] == triples[j]['subjectSpan'][1] \
                        and triples[i]['relationSpan'][0] == triples[j]['relationSpan'][0] \
                        and triples[i]['relationSpan'][1] == triples[j]['relationSpan'][1] \
                        and triples[i]['objectSpan'][0] == triples[j]['objectSpan'][0] \
                        and triples[i]['objectSpan'][1] == triples[j]['objectSpan'][1]:
                    if len('{} {} {}'.format(triples[i]['subject'], triples[i]['relation'], triples[i]['object'])) \
                            <= len('{} {} {}'.format(triples[j]['subject'], triples[j]['relation'], triples[j]['object'])):
                        repeat_mask[i] = 1
                    else:
                        repeat_mask[j] = 1
                if triples[i]['subjectSpan'][0] >= triples[j]['subjectSpan'][0] \
                        and triples[i]['subjectSpan'][1] <= triples[j]['subjectSpan'][1] \
                        and triples[i]['relationSpan'][0] >= triples[j]['relationSpan'][0] \
                        and triples[i]['relationSpan'][1] <= triples[j]['relationSpan'][1] \
                        and triples[i]['objectSpan'][0] >= triples[j]['objectSpan'][0] \
                        and triples[i]['objectSpan'][1] <= triples[j]['objectSpan'][1]:
                    repeat_mask[i] = 1
        return [triples[i] for i in range(len(triples)) if repeat_mask != 1]

    def triples2tensor(self, text):
        # 传入参数 text 中 最多包含 num_triples_per_document 个三元组
        # return: input_ids, attention_mask, start_positions
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:(self.max_seq_len - 2)]

        tokens = ['<s>'] + tokens + ['</s>']
        len_tokens = len(tokens)
        if len_tokens < self.max_seq_len:
            tokens += ['<pad>'] * (self.max_seq_len - len_tokens)

        start_positions = [i for i in range(len(tokens)) if tokens[i] == '<s>']
        if len(start_positions) < self.num_triples_per_document:
            start_positions_mask = [1] * len(start_positions) + [0] * (self.num_triples_per_document - len(start_positions))
            start_positions += [0] * (self.num_triples_per_document - len(start_positions))
        else:
            start_positions_mask = [1] * self.num_triples_per_document

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len_tokens + [0] * (self.max_seq_len - len_tokens)

        return (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(attention_mask, dtype=torch.int64),
            torch.tensor(start_positions, dtype=torch.int64),
            torch.tensor(start_positions_mask, dtype=torch.int64)
        )

    def text2tensor(self, text):
        # return: input_ids, attentions_mask
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > (self.max_seq_len - 2):
            tokens = tokens[:(self.max_seq_len - 2)]
        tokens = ['<s>'] + tokens + ['</s>']
        len_tokens = len(tokens)

        if len_tokens < self.max_seq_len:
            tokens += ['<pad>'] * (self.max_seq_len - len_tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len_tokens + [0] * (self.max_seq_len - len_tokens)

        return (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(attention_mask, dtype=torch.int64)
        )

if __name__ == '__main__':
    tmp = FactDataset('data/tmp.json', 1024, 'facebook/bart-base')
    print(tmp[0])
    print(tmp[1])