from layers import GPT
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics import classification_report
import nltk
import matplotlib.pyplot as plt
import pandas as pd


class GPTDataset(Dataset):
    def __init__(self, raw_text, seq_len):
        super(GPTDataset, self).__init__()
        self.tokens = nltk.word_tokenize(raw_text.lower())

        self.word2idx = self._build_word_2_index()
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        self.seq_len = seq_len
        self.X, self.y = self._build_sequences()

    def _build_word_2_index(self):
        word_2_index = {"<PAD>": 0, "<UNK>": 1}
        for token in self.tokens:
            word_2_index[token] = word_2_index.get(token, len(word_2_index))

        return word_2_index

    def _build_sequences(self):
        X, y = [], []
        if self.seq_len >= len(self.tokens):
            seq = self.tokens[:] + ["<PAD>"] * (self.seq_len - len(self.tokens))
            print(seq)

        else:
            for i in range(len(self.tokens) - self.seq_len):
                seq = self.tokens[i:i + self.seq_len + 1]

                x_tokens = seq[:-1]
                y_tokens = seq[1:]
    
                X.append([self.word2idx[t] for t in x_tokens])
                y.append([self.word2idx[t] for t in y_tokens])

        return torch.tensor(X), torch.tensor(y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":
    test_text = "Hello I am a test sentence."
    gpt_dataset = GPTDataset(test_text, 10)
    # dataloader = DataLoader(gpt_dataset, batch_size=2)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # vocab_size = len(gpt_dataset.word2idx)
    # num_layers = 2
    # seq_len = 4
    # emb_dim = 64
    # n_head = 3
    # epochs = 100
    # learning_rate = 0.0001
    #
    # model = GPT(num_layers, vocab_size, seq_len, emb_dim, n_head).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #
    # for epoch in range(epochs):
    #     epoch_loss = 0
    #     model.train()
    #     for X, y in dataloader:
    #         X, y = X.to(device), y.to(device)
    #
    #         loss, logits = model(X, y)  # logits : (batch_size, seq_len, vocab_size)
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Metrics
    #         epoch_loss += loss.item() * X.size(0)
    #     print("Epoch: {}, Loss: {}".format(epoch, epoch_loss))
    #
    # torch.save(model.state_dict(), "./gpt_weights.pth")
    # out = gpt(test_X.to(device))
    # print(out.size())