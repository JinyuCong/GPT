import json
import re
from collections import Counter
from config import Config as cfg
from datasets import load_dataset


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}

    def _is_word(self, p: str) -> bool:
        return re.fullmatch(r"\w+", p) is not None

    def _tokenize_one(self, word: str):
        tokens = list(word) + ['</w>']

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

        if tokens and tokens[-1] == '</w>':
            tokens = tokens[:-1]

        return tokens

    def tokenize(self, text: str, return_tokens=False):
        pieces = re.findall(r"\w+|[^\w\s]", text.lower())
        output_tokens = []

        for p in pieces:
            # 如果是标点，直接当 token
            if self._is_word(p):
                output_tokens.extend(self._tokenize_one(p))
            else:
                output_tokens.append(p)

        token_ids = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in output_tokens]
        if return_tokens:
            return token_ids, output_tokens
        else:
            return token_ids

    def train(self, train_text: str, max_concat_num: int, save_logs=True):
        # 将训练数据拆分成char
        pieces = re.findall(r"\w+|[^\w\s]", train_text.lower())

        # 统计词频
        word_freq = Counter(pieces)

        # corpus: token序列(tuple) -> count
        corpus = {}
        for w, cnt in word_freq.items():
            if self._is_word(w):
                tokens = tuple(list(w) + ['</w>'])
            else:
                tokens = (w,)  # 标点情况
            corpus[tokens] = cnt

        # 迭代合并
        for _ in range(max_concat_num):
            bigram_freqs = Counter()

            # 找到最频繁的bigram
            for tokens, cnt in corpus.items():
                for i in range(len(tokens) - 1):
                    bigram_freqs[(tokens[i], tokens[i + 1])] += cnt

            if not bigram_freqs:
                break

            (a, b), freq = bigram_freqs.most_common(1)[0]
            if freq < 2:
                break
            self.merges.append((a, b))

            # 合并最频繁的bigram
            new_corpus = {}
            for tokens, cnt in corpus.items():
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i] == a) and (tokens[i + 1] == b):
                        new_tokens.append(a + b)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_corpus[tuple(new_tokens)] = new_corpus.get(tuple(new_tokens), 0) + cnt

            # 更新char level训练数据
            corpus = new_corpus

        for tokens in corpus.keys():
            for t in tokens:
                if t == "</w>":
                    continue
                if t not in self.word2idx:
                    self.word2idx[t] = len(self.word2idx)

        logs = {
            "word2idx": self.word2idx,
            "merges": self.merges,
        }

        if save_logs:
            with open('logs.json', "w", encoding="utf-8") as f:
                json.dump(logs, f)

    def from_pretrained(self, log_path: str):
        with open(log_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        word2idx, merges = content['word2idx'], content['merges']

        self.word2idx = word2idx
        self.merges = merges
        

if __name__ == '__main__':
    # 训练tokenizer
    with open("./Alan_Turing_corpus.txt", "r", encoding="utf-8") as f:
        train_text = f.read()

    tokenizer = BPETokenizer()
    tokenizer.train(train_text, cfg.max_concat_num, save_logs=True)

