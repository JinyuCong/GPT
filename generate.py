import os
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from layers import GPT
from config import Config
from dataset import GPTDataset
import nltk
import json


@torch.no_grad()
def generate(
        model, input_ids, max_new_tokens, device,
        block_size, temperature=0.7, top_k=50
):
    model.eval()

    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)

    input_ids = input_ids.to(device)

    for _ in range(max_new_tokens):
        x = input_ids[-block_size:].unsqueeze(0)
        logits = model(x)
        next_token_logits = logits[:, -1, :]  # (1, V)

        next_token_logits = next_token_logits / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            # 挑出前k个最高概率的token
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))  # V: (1, top_k)
            cutoff = v[:, -1].unsqueeze(-1)
            # 给所有的下一个token的概率做裁剪，如果概率小于topk最小token概率则变为负无穷（永远不选），如果大于则还是原始概率
            next_token_logits = torch.where(next_token_logits < cutoff, torch.tensor(float('-inf'), device=device), next_token_logits)

        # 转换为概率
        probs = torch.softmax(next_token_logits, dim=-1)  # (1, V)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        input_ids = torch.concat([input_ids, next_token], dim=0)
        decoded = decode(input_ids, idx2word)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(decoded)

    return input_ids


def decode(input_ids: torch.tensor, idx2word: dict) -> str:
    return " ".join(idx2word[int(i)] for i in input_ids)


if __name__ == '__main__':
    with open("./test_data.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    cfg = Config

    with open("./word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(word2idx)

    model = GPT(cfg.num_layers, vocab_size, cfg.seq_len, cfg.emb_dim, cfg.n_head).to(cfg.device)
    model.load_state_dict(torch.load('./gpt_weights.pth', weights_only=True))

    test = "because features"
    tokens = nltk.word_tokenize(test.lower())
    tokens_ids = [word2idx[t] for t in tokens]

    generated = generate(
        model=model,
        input_ids=tokens_ids,
        max_new_tokens=100,
        device=cfg.device,
        block_size=cfg.seq_len,
        temperature=cfg.temperature,
        top_k=cfg.top_k
    )

