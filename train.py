import os
import torch
import argparse

from torch.utils.data import DataLoader
from model.factBart import FactBartForGeneration
from utils.bart_dataset import FactDataset
from utils.trainer import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup


def train(args):
    train_dataset = FactDataset(data_path=os.path.join(args.data_dir, 'train.json'), max_seq_len=args.max_position_embeddings, tokenizer_path=args.tokenizer_name)
    # valid_dataset = FactDataset(data_path=os.path.join(args.data_dir, 'valid.json'), max_seq_len=args.max_position_embeddings, tokenizer_path=args.tokenizer_name)
    # test_dataset = FactDataset(data_path=os.path.join(args.data_dir, 'test.json'), max_seq_len=args.max_position_embeddings, tokenizer_path=args.tokenizer_name)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.train_shuffle
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FactBartForGeneration(args)

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=int(args.warmup_rate * len(train_loader))
    )

    trainer = Trainer(
        model=model,
        device=device,
        model_dir=args.model_saved_dir,
        tokenizer_name=args.tokenizer_name
    )

    trainer.train(
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        print_per_iter=args.print_per_iter
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir', default='data/', type=str)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-train_shuffle', default=True, type=bool)
    parser.add_argument('-learning_rate', default=5e-5, type=float)
    parser.add_argument('-warmup_rate', default=0.1, type=float)
    parser.add_argument('-model_saved_dir', default='./results/saving_models', type=str)
    parser.add_argument('-tokenizer_name', default='./facebook/bart-base', type=str)
    parser.add_argument('-epochs', default=10, type=int)
    parser.add_argument('-print_per_iter', default=1000, type=int)

    parser.add_argument('-d_model', default=768, type=int)
    parser.add_argument('-vocab_size', default=50265, type=int)
    parser.add_argument('-pad_token_id', default=1, type=int)
    parser.add_argument('-decoder_start_token_id', default=2, type=int)
    parser.add_argument('-output_attentions', default=False, type=bool)
    parser.add_argument('-output_hidden_states', default=False, type=bool)
    parser.add_argument('-use_cache', default=True, type=bool)
    parser.add_argument('-dropout', default=0.1, type=float)
    parser.add_argument('-encoder_layerdrop', default=0.0, type=float)
    parser.add_argument('-decoder_layerdrop', default=0.0, type=float)
    parser.add_argument('-max_position_embeddings', default=1024, type=int)
    parser.add_argument('-num_encoder_layers', default=6, type=int)
    parser.add_argument('-num_decoder_layers', default=6, type=int)
    parser.add_argument('-decoder_num_attention_heads', default=12, type=int)
    parser.add_argument('-encoder_num_attention_heads', default=12, type=int)
    parser.add_argument('-attention_dropout', default=0.1, type=float)
    parser.add_argument('-encoder_ffn_dim', default=3072, type=int)
    parser.add_argument('-decoder_ffn_dim', default=3072, type=int)
    parser.add_argument('-activation_dropout', default=0.1, type=float)

    args = parser.parse_args()
    train(args)