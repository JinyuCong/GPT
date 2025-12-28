from torch.utils.data import Dataset
from tokenizer import BPETokenizer
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from config import Config as cfg


class GPTDataset(Dataset):
    def __init__(self, ds_name, seq_len):
        super(GPTDataset, self).__init__()
        self.raw_data = self._load_raw_data(ds_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.X, self.y = self._build_sequences()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    def _load_raw_data(self, ds_name: str):
        raw_data = load_dataset(ds_name)
        return raw_data["train"]["text"]

    def _build_sequences(self):
        all_tokens = []
        X, y = [], []
        for raw_text in self.raw_data:
            all_tokens.extend(self.tokenizer.encode(raw_text))
            all_tokens.append(self.tokenizer.eos_token_id)

        for i in range(len(all_tokens) - self.seq_len):
            seq = all_tokens[i:i + self.seq_len + 1]

            x_tokens = seq[:-1]
            y_tokens = seq[1:]

            X.append(x_tokens)
            y.append(y_tokens)

        return torch.tensor(X), torch.tensor(y)


if __name__ == "__main__":
    dataset = GPTDataset("tatsu-lab/alpaca", seq_len=cfg.seq_len)
    print(dataset[0])
