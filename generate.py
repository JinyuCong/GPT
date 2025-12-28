import os
import re

import torch
import torch.nn.functional as F
import torch.nn as nn
from layers import GPT
from config import Config as cfg
from dataset import GPTDataset
from tokenizer import BPETokenizer
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

    with open("./logs.json", "r", encoding="utf-8") as f:
        logs = json.load(f)

    tokenizer = BPETokenizer(save_logs=False)
    tokenizer.from_pretrained("./logs.json")

    word2idx = tokenizer.word2idx
    idx2word = {idx: re.sub("</w>", "", word) for word, idx in word2idx.items()}
    vocab_size = len(word2idx)

    model = GPT(cfg.num_layers, vocab_size, cfg.seq_len, cfg.emb_dim, cfg.n_head).to(cfg.device)
    model.load_state_dict(torch.load('./gpt_weights.pth', weights_only=True))

    test = "turing was raised"
    tokens_ids = tokenizer.tokenize(test)

    generate_ids = generate(
        model=model,
        input_ids=tokens_ids,
        max_new_tokens=cfg.max_new_tokens,
        device=cfg.device,
        block_size=cfg.seq_len,
        temperature=cfg.temperature,
        top_k=cfg.top_k
    )

    # decoded = decode(generate_ids, idx2word)
    # print(decoded)

