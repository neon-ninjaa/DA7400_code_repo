import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, embedding_dim, hidden_dim, seq_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z):
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        outputs = []
        input_token = torch.zeros(z.size(0), 1, dtype=torch.long, device=z.device)

        for _ in range(self.seq_len):
            embedded = self.embedding(input_token)
            lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            logits = self.fc(lstm_out.squeeze(1))
            input_token = logits.argmax(dim=1).unsqueeze(1)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)
