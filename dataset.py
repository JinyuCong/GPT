from torch.utils.data import Dataset, DataLoader
import nltk
import torch


class GPTDataset(Dataset):
    def __init__(self, raw_text, seq_len):
        super(GPTDataset, self).__init__()
        self.tokens = nltk.word_tokenize(raw_text.lower())

        self.word2idx = self._build_word_2_index()
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        self.vocab_size = len(self.word2idx)
        self.seq_len = seq_len
        self.X, self.y = self._build_sequences()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

    def _build_word_2_index(self):
        word_2_index = {"<PAD>": 0, "<UNK>": 1}
        for token in self.tokens:
            word_2_index[token] = word_2_index.get(token, len(word_2_index))

        return word_2_index

    def _build_sequences(self):
        X, y = [], []
        if self.seq_len >= len(self.tokens):
            raise ValueError(
                f"Number of token in dataset too less, "
                f"number of tokens({len(self.tokens)}) < defined sequence length({self.seq_len}). "
                f"Needs to reduce the sequence length."
            )

        else:
            for i in range(len(self.tokens) - self.seq_len):
                seq = self.tokens[i:i + self.seq_len + 1]

                x_tokens = seq[:-1]
                y_tokens = seq[1:]

                X.append([self.word2idx[t] for t in x_tokens])
                y.append([self.word2idx[t] for t in y_tokens])

        return torch.tensor(X), torch.tensor(y)
