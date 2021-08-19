import json
from tqdm import tqdm
from rouge import Rouge
from transformers import AutoTokenizer

'''
    1. article triples 去重：
    2. 使用 rouge 获取 triples mask，某个排名阈值并且 rouge 分数小于阈值的为0，其他为 1
    3. 拼接 article triples
    4. 获取 article，highlights，article triples的词向量，外加article triples中 <s> 的位置
'''


def preprocess(data_file, save_path, tokenizer, max_seq_len=1024, num_triples_per_document=64,
               num_target_triples_per_document=16):
    rouge = Rouge()
    with open(data_file, 'r') as freader, open(save_path, 'w') as fwriter:
        for line in tqdm(freader.readlines(), desc=save_path):
            line = json.loads(line)
            id = line['id']
            article = line['article']
            article_triples = line['article_triples']
            highlights = line['highlights']
            highlights_triples = line['highlights_triples']

            # article_triples 去重，type = list
            # DO IT IN PREPROCESS
            # article_triples = self.remove_repeat_triple(article_triples)

            # 获取 article triple 在 highlights 中的 rouge
            # type(article_triples) = str
            # type(article_triples_label) = torch.Tensor(list)
            # type(article_triples_label_mask) = torch.Tensor(list)
            # TODO self.num_target_triples_per_document 代码内聚性和耦合性可优化
            article_triples, article_triples_label, article_triples_label_mask = get_top_k_triple(
                article_triples=article_triples,
                highlights=highlights,
                rouge=rouge,
                k_num=num_target_triples_per_document,
                rule='l',
                threshold=1,
                num_triples_per_document=num_triples_per_document
            )

            article_input_ids, article_attention_mask = text2id(
                text=article,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len
            )
            highlights_input_ids, highlights_attention_mask = text2id(
                text=highlights,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len
            )

            article_triples_input_ids, article_triples_attention_mask, article_triples_start_positions, article_triples_start_positions_mask = triples2id(
                text=article_triples,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                num_triples_per_document=num_triples_per_document
            )

            # TODO: article_triple_label 中 CLS 个数大于 article_triples_s_positions 中 CLS 的情况
            # 要根据 article_triples_s_positions 的数量标记 article_triples_start 值的有效位
            # triple_hidden_states + article_triple_s_positions + article_triples_attention_mask
            # => document 中所有三元组的CLS
            assert len(
                article_triples_start_positions) == num_triples_per_document, 'triples_start_positions != num_triples'
            assert len(
                article_triples_start_positions_mask) == num_triples_per_document, 'triples_start_positions_mask != num_triples'
            assert len(article_triples_label) == num_triples_per_document, 'triples_label != num_triples'
            assert len(article_triples_label_mask) == num_triples_per_document, 'triples_label_mask != num_triples'

            fwriter.write(json.dumps(
                {
                    'id': id,
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
                }
            ) + '\n')


def get_top_k_triple(article_triples, highlights, rouge, k_num=5, rule='l', threshold=1, num_triples_per_document=64):
    # threshold 默认值为1
    ts = []

    for triple in article_triples:
        subj = triple['subject']
        rel = triple['relation']
        obj = triple['object']
        triple = '{} {} {}'.format(subj, rel, obj)
        scores = rouge.get_scores(triple, highlights)
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
    ts = ts[:num_triples_per_document]  # 每一个 document 中三元组的个数
    triple_tgt_label = [1] * len(ts) + [0] * (num_triples_per_document - len(ts))

    for i in range(k_num, min(len(article_triples), num_triples_per_document)):

        score = ts[i][1] if (rule == '1' or rule == 'rouge-1') else (
            ts[i][2] if (rule == '2' or rule == 'rouge-2') else ts[i][3])
        if score < threshold:
            triple_tgt_label[i] = 0

    # assert len(ts) == len(triple_tgt_label), 'length of triples must equal to length of mask'
    # triple_tgt_label_mask 表示 document 中三元组的个数
    triple_tgt_label_mask = [1] * len(ts) + [0] * (num_triples_per_document - len(ts))
    sorted_ts = [t[0] for t in ts]
    return (
        '<s>'.join(sorted_ts),
        triple_tgt_label,
        triple_tgt_label_mask
    )


def triples2id(text, tokenizer, max_seq_len, num_triples_per_document):
    # 传入参数 text 中 最多包含 num_triples_per_document 个三元组
    # return: input_ids, attention_mask, start_positions
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[:(max_seq_len - 2)]

    tokens = ['<s>'] + tokens + ['</s>']
    len_tokens = len(tokens)
    if len_tokens < max_seq_len:
        tokens += ['<pad>'] * (max_seq_len - len_tokens)

    start_positions = [i for i in range(len(tokens)) if tokens[i] == '<s>']
    if len(start_positions) < num_triples_per_document:
        start_positions_mask = [1] * len(start_positions) + [0] * (num_triples_per_document - len(start_positions))
        start_positions += [0] * (num_triples_per_document - len(start_positions))
    else:
        start_positions_mask = [1] * num_triples_per_document

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    attention_mask = [1] * len_tokens + [0] * (max_seq_len - len_tokens)

    return (
        input_ids,
        attention_mask,
        start_positions,
        start_positions_mask
    )


def text2id(text, tokenizer, max_seq_len):
    # return: input_ids, attentions_mask
    tokens = tokenizer.tokenize(text)
    if len(tokens) > (max_seq_len - 2):
        tokens = tokens[:(max_seq_len - 2)]
    tokens = ['<s>'] + tokens + ['</s>']
    len_tokens = len(tokens)

    if len_tokens < max_seq_len:
        tokens += ['<pad>'] * (max_seq_len - len_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len_tokens + [0] * (max_seq_len - len_tokens)

    return (
        input_ids,
        attention_mask
    )


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('../pretrained_models/bart_base/')
    # preprocess(
    #     data_file='./origin_data/cnn_dailymail/train.json',
    #     save_path='./data/cnn_dailymail/train.json',
    #     tokenizer=tokenizer,
    #     max_seq_len=1024,
    #     num_triples_per_document=16,
    #     num_target_triples_per_document=8
    # )

    # preprocess(
    #     data_file='./origin_data/cnn_dailymail/valid.json',
    #     save_path='./data/cnn_dailymail/valid.json',
    #     tokenizer=tokenizer,
    #     max_seq_len=1024,
    #     num_triples_per_document=16,
    #     num_target_triples_per_document=8
    # )

    # preprocess(
    #     data_file='./origin_data/cnn_dailymail/test.json',
    #     save_path='./data/cnn_dailymail/test.json',
    #     tokenizer=tokenizer,
    #     max_seq_len=1024,
    #     num_triples_per_document=16,
    #     num_target_triples_per_document=8
    # )

    preprocess(
        data_file='./origin_data/xsum/train.json',
        save_path='./data/xsum/train.json',
        tokenizer=tokenizer,
        max_seq_len=1024,
        num_triples_per_document=16,
        num_target_triples_per_document=8
    )

    preprocess(
        data_file='./origin_data/xsum/valid.json',
        save_path='./data/xsum/valid.json',
        tokenizer=tokenizer,
        max_seq_len=1024,
        num_triples_per_document=16,
        num_target_triples_per_document=8
    )

    preprocess(
        data_file='./origin_data/xsum/test.json',
        save_path='./data/xsum/test.json',
        tokenizer=tokenizer,
        max_seq_len=1024,
        num_triples_per_document=16,
        num_target_triples_per_document=8
    )