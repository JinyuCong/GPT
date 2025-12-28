from layers import GPT
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config as cfg
from dataset import GPTDataset
from transformers import AutoTokenizer


class EarlyStopping:
    def __init__(self, model, verbose=True, patience=2, delta=0):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.patience_counter = 0
        self.min_loss = torch.inf
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, loss):
        # in the case of validation accuracy do not improve
        if loss > self.min_loss + self.delta:
            self.patience_counter += 1
            if self.verbose:
                print(f"Validation accuracy did not improve. Patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                self.early_stop = True
        else:
            self.min_loss = loss
            self.patience_counter = 0
            self.best_model_weights = self.model.state_dict().copy()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    gpt_dataset = GPTDataset(cfg.dataset_name, cfg.seq_len)
    dataloader = DataLoader(gpt_dataset, batch_size=cfg.batch_size)
    num_steps = len(dataloader)
    print(f"INFO: Dataloader initialized successfully. Number of training steps: {num_steps}")

    vocab_size = tokenizer.vocab_size

    model = GPT(cfg.num_layers, vocab_size, cfg.seq_len, cfg.emb_dim, cfg.n_head).to(cfg.device)
    print(f"INFO: Model initialized successfully")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    early_stopping = EarlyStopping(model)

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

        epoch_mean_loss = epoch_loss / num_steps
        early_stopping(epoch_mean_loss)

        if early_stopping.early_stop:
            print(f"Early stopping triggered. Best model weight saved at ./gpt_weights.pth")
            torch.save(early_stopping.best_model_weights, "./gpt_weights.pth")
            break

        print(f"Epoch: {epoch+1}/{cfg.epochs}, Loss: {epoch_mean_loss:.4f}, minimum loss: {early_stopping.min_loss:.4f}")
        print("-----------------------------------")

    torch.save(model.state_dict(), "./gpt_weights.pth")

    # test_X = gpt_dataset[0][0].unsqueeze(0).to(cfg.device)  # (1, T, V)
    # logits = model(test_X)  # (1, T, V)
    # next_token_logits = logits[:, -1, :]  # (1, V)
    # next_token = torch.argmax(next_token_logits, dim=-1)  # (1,)
    # print(next_token)

