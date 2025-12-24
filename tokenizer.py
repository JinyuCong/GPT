import re
from collections import Counter
import torch


class BPETokenizer:
    def __init__(self, train_text: str, max_concat_num: int):
        self.train_text = train_text
        self.max_concat_num = max_concat_num
        self.merges = []
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}

    def _tokenize_one(self, word: str):
        tokens = list(word.lower()) + ['</w>']

        for a, b in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        if tokens[-1] == '</w>':
            tokens = tokens[:-1]

        return tokens

    def tokenize(self, text):
        pieces = re.findall(r"\w+|[^\w\s]", text)
        output_tokens = []

        for p in pieces:
            # 如果是标点，直接当 token
            if not p.isalnum():
                output_tokens.append(p)
            else:
                output_tokens.extend(
                    self._tokenize_one(p)
                )

        token_ids = []
        for t in output_tokens:
            if t in self.word2idx:
                token_ids.append(self.word2idx[t])
            else:
                token_ids.append(self.word2idx['<UNK>'])

        return torch.tensor(token_ids)

    def train(self):
        # 将训练数据拆分成char
        words = re.findall(r"\w+|[^\w\s]", self.train_text)
        corpus = [[c.lower() for c in word] + ['</w>'] for word in words]

        for _ in range(self.max_concat_num):
            bigram_freqs = Counter()
            # 找到最频繁的bigram
            for word in corpus:
                for i in range(len(word) - 1):
                    bigram_freqs[(word[i], word[i + 1])] += 1

            if not bigram_freqs:
                break

            (a, b), freq = bigram_freqs.most_common(1)[0]
            self.merges.append((a, b))

            if freq < 2:
                break

            # 合并最频繁的bigram
            new_corpus = []
            for word in corpus:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i] == a) and (word[i + 1] == b):
                        new_word.append(a + b)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_corpus.append(new_word)

            # 更新char level训练数据
            corpus = new_corpus

        for word in corpus:
            for token in word:
                if token not in self.word2idx:
                    self.word2idx[token] = len(self.word2idx)
        

if __name__ == '__main__':
    train_text = '''
    Natural language processing is a core topic in machine learning.
    In natural language processing, tokenization plays a crucial role in how text is represented.
    Traditional word-based tokenization relies on fixed vocabularies, which often fail when new words appear.
    
    Subword methods such as byte pair encoding, or BPE, address this issue by learning frequent patterns from data.
    The BPE algorithm starts with characters and repeatedly merges the most frequent adjacent pairs.
    Over time, common sequences like language, processing, and tokenization become single tokens.
    
    In practice, a BPE tokenizer balances vocabulary size and expressive power.
    It can represent common words efficiently while still handling rare words, typos, and unseen forms.
    This flexibility is one reason BPE-style tokenizers are widely used in modern language models.
    
    Version 1.0 and version 2.0 of a tokenizer may behave differently.
    Testing on examples like “tokenize”, “tokenized”, and “tokenization” helps reveal how merges are learned.
    '''

    tokenizer = BPETokenizer(train_text, 200)
    tokenizer.train()
    print(tokenizer.tokenize("natural language process is a "))

    test_text = "natural language process is"
    word2idx = {"natural": 0, "language": 1, "process": 2, "ing": 3, "tokenization": 4}
