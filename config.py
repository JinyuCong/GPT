from dataclasses import dataclass
import torch


@dataclass
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    seq_len = 64
    emb_dim = 256
    num_layers = 12
    n_head = 3
    epochs = 100
    learning_rate = 1e-4
    temperature = 1
    top_k = 100
