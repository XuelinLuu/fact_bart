import torch
from torch import nn

class BartLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids)
