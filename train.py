from layers import GPT
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from config import Config
from dataset import GPTDataset
import json


if __name__ == "__main__":
    with open("./test_data.txt", "r", encoding="utf-8") as f:
        test_text = f.read()

    cfg = Config
    gpt_dataset = GPTDataset(test_text, cfg.seq_len)
    with open("./word2idx.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(gpt_dataset.word2idx))

    dataloader = DataLoader(gpt_dataset, batch_size=cfg.batch_size)

    vocab_size = len(gpt_dataset.word2idx)

    model = GPT(cfg.num_layers, vocab_size, cfg.seq_len, cfg.emb_dim, cfg.n_head).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        epoch_loss = 0
        num_step = 0
        model.train()
        for X, y in dataloader:
            X, y = X.to(cfg.device), y.to(cfg.device)

            step_loss, logits = model(X, y)  # logits : (batch_size, seq_len, vocab_size)

            # Backward pass
            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1}/{cfg.epochs}, Step: {num_step}/{len(dataloader)}, Step loss: {step_loss:.4f}")

            # Metrics
            epoch_loss += step_loss.item()
            num_step += 1
        print(f"Epoch: {epoch+1}/{cfg.epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        print("-----------------------------------")

    torch.save(model.state_dict(), "./gpt_weights.pth")

    # test_X = gpt_dataset[0][0].unsqueeze(0).to(cfg.device)  # (1, T, V)
    # logits = model(test_X)  # (1, T, V)
    # next_token_logits = logits[:, -1, :]  # (1, V)
    # next_token = torch.argmax(next_token_logits, dim=-1)  # (1,)
    # print(next_token)

