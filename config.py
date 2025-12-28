from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Training
    dataset_name = "tatsu-lab/alpaca"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    seq_len = 64
    emb_dim = 256
    num_layers = 12
    n_head = 3
    epochs = 30
    learning_rate = 1e-4

    # Generation
    temperature = 1.5
    top_k = 1000
    max_new_tokens = 200

    # Tokenizer
    tokenizer_name = "meta-llama/Meta-Llama-3-8B"
    max_concat_num = 3000
