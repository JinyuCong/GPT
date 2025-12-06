import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
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


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, emb_dim):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, emb_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, seq_len, emb_dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_head, dk, dv):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.dk = dk
        self.dv = dv

        self.wq = nn.Linear(emb_dim, n_head * dk).to(device)
        self.wk = nn.Linear(emb_dim, n_head * dk).to(device)
        self.wv = nn.Linear(emb_dim, n_head * dv).to(device)

        self.wo = nn.Linear(n_head * dv, emb_dim).to(device)

    def forward(self, x_emb: torch.tensor):
        batch_size, seq_len, emb_dim = x_emb.shape

        Q = self.wq(x_emb).view(batch_size, seq_len, self.n_head, self.dk)  # (batch_size, seq_len, n_head, dk)
        K = self.wk(x_emb).view(batch_size, seq_len, self.n_head, self.dk)
        V = self.wv(x_emb).view(batch_size, seq_len, self.n_head, self.dv)

        Q = Q.permute(0, 2, 1, 3)  # (batch_size, n_head, seq_len, dk)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)  # (batch_size, n_head, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, n_head, seq_len, dv)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, n_head, dv)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch_size, seq_len, n_head * dv)
        attn_output = self.wo(attn_output)  # (batch_size, seq_len, emb_dim)

        return attn_output


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_head, dk, dv):
        super(MaskedMultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.dk = dk
        self.dv = dv

        self.wq = nn.Linear(emb_dim, n_head * dk).to(device)
        self.wk = nn.Linear(emb_dim, n_head * dk).to(device)
        self.wv = nn.Linear(emb_dim, n_head * dv).to(device)

        self.wo = nn.Linear(n_head * dv, emb_dim).to(device)

    def forward(self, x_emb: torch.tensor):
        batch_size, seq_len, emb_dim = x_emb.shape

        Q = self.wq(x_emb).view(batch_size, seq_len, self.n_head, self.dk)  # (batch_size, seq_len, n_head, dk)
        K = self.wk(x_emb).view(batch_size, seq_len, self.n_head, self.dk)
        V = self.wv(x_emb).view(batch_size, seq_len, self.n_head, self.dv)

        Q = Q.permute(0, 2, 1, 3)  # (batch_size, n_head, seq_len, dk)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        mask = torch.tril(torch.ones((seq_len, seq_len))).to(device)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)  # (batch_size, n_head, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(mask == 0, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, n_head, seq_len, dv)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, n_head, dv)
        attn_output = attn_output.view(batch_size, seq_len, -1)  # (batch_size, seq_len, n_head * dv)
        attn_output = self.wo(attn_output)  # (batch_size, seq_len, emb_dim)

        return attn_output


class FeedForward(nn.Module):
    def __init__(self, emb_dim, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(emb_dim, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout=0.1):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor, sublayer_out: torch.tensor):
        return self.layer_norm(x + self.dropout(sublayer_out))


class DecoderLayer(nn.Module):
    def __init__(self, n_head, emb_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(emb_dim, n_head, emb_dim // n_head, emb_dim // n_head)
        self.masked_attn = MaskedMultiHeadAttention(emb_dim, n_head, emb_dim // n_head, emb_dim // n_head)

        self.ff = FeedForward(emb_dim, 4*emb_dim, dropout)

        self.add_norm1 = AddNorm(emb_dim, dropout)
        self.add_norm2 = AddNorm(emb_dim, dropout)
        self.add_norm3 = AddNorm(emb_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_emb: torch.tensor):
        masked_attn_out = self.masked_attn(x_emb)
        add_norm_masked_attn = self.add_norm1(x_emb, masked_attn_out)

        attn_out = self.attn(add_norm_masked_attn)
        add_norm_attn = self.add_norm2(add_norm_masked_attn, attn_out)

        ff_out = self.ff(add_norm_attn)
        add_norm_ff = self.add_norm3(add_norm_attn, ff_out)

        return add_norm_ff


class GPT(nn.Module):
    def __init__(self, num_layers, vocab_size, seq_len, emb_dim, n_head):
        super(GPT, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(seq_len, emb_dim)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(n_head, emb_dim) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding_layer.weight

    def forward(self, x, labels=None):
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)

        for decoder in self.decoder_layers:
            x = decoder(x)
        logits = self.lm_head(x)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab)
                labels.view(-1)  # (batch_size * seq_len, )
            )
            return loss, logits

        return logits


if __name__ == "__main__":
    test_text = "Hello I am a test sentence."
    gpt_dataset = GPTDataset(test_text, 4)
    dataloader = DataLoader(gpt_dataset, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(gpt_dataset.word2idx)
    num_layers = 2
    seq_len = 4
    emb_dim = 64
    n_head = 3
    epochs = 100
    learning_rate = 0.0001

    model = GPT(num_layers, vocab_size, seq_len, emb_dim, n_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            loss, logits = model(X, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item() * X.size(0)
        print("Epoch: {}, Loss: {}".format(epoch, epoch_loss))

    # out = gpt(test_X.to(device))
    # print(out.size())
