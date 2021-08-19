import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from rouge import Rouge
from tqdm import tqdm
from torchsummary import summary


class Trainer():
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            model_dir,
            tokenizer_name='facebook/bart-base'
    ):
        self.model = model
        self.device = device
        self.model_dir = model_dir
        self.rouge = Rouge()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def saving_model(self, model_name):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, model_name))

    def loading_model(self, model_name):
        if not os.path.exists(model_name):
            raise ValueError('{} is not exist'.format(model_name))
        self.model.load_state_dict(torch.load(model_name))

    def get_evaluation(self, y_true, y_pred):
        p_macro = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
        p_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')

        r_macro = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
        r_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')

        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

        return {
            'macro': {
                'p': p_macro,
                'r': r_macro,
                'f1': f1_macro
            },
            'micro': {
                'p': p_micro,
                'r': r_micro,
                'f1': f1_micro
            }
        }

    def print_rank(self, evaluation):
        return 'macro->【p: {} r: {} f1: {}】 micro->【p: {} r: {} f1: {}】'.format(
            evaluation['macro']['p'],
            evaluation['macro']['r'],
            evaluation['macro']['f1'],
            evaluation['micro']['p'],
            evaluation['micro']['r'],
            evaluation['micro']['f1'],
        )

    def print_rank_matrix(self, evaluation):
        print('{}\t{}\t{}\t{}\n{}\t{}\t{}\t{}\n{}\t{}\t{}\t{}\n'.format(
            'macro', 'p', 'r', 'f1',
            'macro', evaluation['macro']['p'], evaluation['macro']['r'], evaluation['macro']['f1'],
            'micro', evaluation['micro']['p'], evaluation['micro']['r'], evaluation['micro']['f1'],
        ))

    def get_rouge(self, y_true, y_pred):
        text_true = self.tokenizer.decode(y_true)
        text_pred = self.tokenizer.decode(y_pred)
        # TODO

    def train(self, dataloader: DataLoader, optimizer, scheduler, epochs=10, print_per_iter=1000):
        self.model.train()
        # self.model.to(self.device)
        for epoch in range(epochs):
            for idx, batch in enumerate(dataloader):

                # print(self.model)
                #
                # summary(self.model, input_size=[
                #     batch[0].size(), batch[1].size(), batch[2].size(), batch[3].size(), batch[4].size(),
                #     batch[5].size(), \
                #     batch[6].size(), batch[7].size(), batch[8].size(), batch[9].size()
                # ])
                #
                # exit()

                article_input_ids, article_attention_mask, highlights_input_ids, highlights_attention_mask, \
                article_triples_input_ids, article_triples_attention_mask, article_triples_start_positions, \
                article_triples_start_positions_mask, article_triples_label, article_triples_label_mask = \
                    batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), \
                    batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), \
                    batch[8].cuda(), batch[9].cuda()

                optimizer.zero_grad()
                output = self.model(
                    article_input_ids=article_input_ids,  # input_ids
                    article_attention_mask=article_attention_mask,  # attention_mask
                    highlights_input_ids=highlights_input_ids,  # decoder_input_ids
                    highlights_attention_mask=highlights_attention_mask,  # decoder_attention_mask
                    article_triples_input_ids=article_triples_input_ids,
                    article_triples_attention_mask=article_triples_attention_mask,
                    article_triples_start_positions=article_triples_start_positions,
                    article_triples_start_positions_mask=article_triples_start_positions_mask,
                    article_triples_label=article_triples_label,
                    article_triples_label_mask=article_triples_label_mask
                )
                y_pred = torch.argmax(output[1], dim=-1).detach().cpu().numpy()
                y_true = highlights_input_ids.detach().cpu().numpy()
                # evaluation = self.get_evaluation(y_true=y_true, y_pred=y_pred)

                loss = output[0]
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if ((epoch - 1) * len(dataloader) + idx) % print_per_iter == 0 or idx == len(dataloader) - 1:
                    inform = 'epoch_{}_iter_{}_loss_{}'.format(
                        epoch, idx, loss.item()
                    )
                    print(inform)
                    self.saving_model(inform + '.bin')

    def eval(self, dataloader, model_name=None):
        if model_name is None:
            names = os.listdir(self.model_dir)
            model_name = sorted(names)[-1]
        self.loading_model(model_name)
        # self.model.to(self.device)
        self.model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                article_input_ids, article_attention_mask, highlights_input_ids, highlights_attention_mask, \
                article_triples_input_ids, article_triples_attention_mask, article_triples_start_positions, \
                article_triples_start_positions_mask, article_triples_label, article_triples_label_mask = \
                    batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), \
                    batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda(), \
                    batch[8].cuda(), batch[9].cuda()

                output = self.model(
                    article_input_ids=article_input_ids,  # input_ids
                    article_attention_mask=article_attention_mask,  # attention_mask
                    highlights_input_ids=highlights_input_ids,  # decoder_input_ids
                    highlights_attention_mask=highlights_attention_mask,  # decoder_attention_mask
                    article_triples_input_ids=article_triples_input_ids,
                    article_triples_attention_mask=article_triples_attention_mask,
                    article_triples_start_positions=article_triples_start_positions,
                    article_triples_start_positions_mask=article_triples_start_positions_mask,
                )
                y_pred = torch.argmax(output[0], dim=-1).detach().cpu().numpy()
                y_true = highlights_input_ids.detach().cpu().numpy()

                trues.extend(list(y_true))
                preds.extend(list(y_pred))
        # evaluation = self.get_evaluation(y_true=trues, y_pred=preds)
        # self.print_rank_matrix(evaluation)
