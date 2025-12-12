from layers import GPT, GPTDataset
import torch
from transformers import AutoTokenizer, AutoModel
import nltk


if __name__ == '__main__':
    test_text = "Hello I am a test sentence."
    gpt_dataset = GPTDataset(test_text, 4)

    lala = "Hello I am a"
    lala_tokens = nltk.word_tokenize(lala.lower())
    encoded_lala = torch.tensor([[gpt_dataset.word2idx[token] for token in lala_tokens]])

    vocab_size = len(gpt_dataset.word2idx)
    num_layers = 2
    seq_len = 4
    emb_dim = 64
    n_head = 3
    epochs = 100
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoded_lala = encoded_lala
    model = GPT(num_layers, vocab_size, seq_len, emb_dim, n_head)
    model.load_state_dict(torch.load("./gpt_weights.pth"))

    model.eval()
    logits = model(encoded_lala)

    print(gpt_dataset.word2idx)
    next_token = logits[:, -1, :].argmax(dim=1)
    print(gpt_dataset.word2idx[next_token.item()])
