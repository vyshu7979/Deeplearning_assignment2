
# seq2seq_transliteration.py (Fixed tensor size mismatch in collate_fn)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from google.colab import drive
import numpy as np
import os
from google.colab import files
from sklearn.metrics import accuracy_score
from Levenshtein import distance

# ----------------------- CONFIG -----------------------
class Config:
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 1
    cell_type = 'LSTM'
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    teacher_forcing_ratio = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    sos_token = '<SOS>'
    eos_token = '<EOS>'

# ----------------------- SEEDING -----------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.seed)

# ----------------------- DATA -----------------------
class TransliterationDataset(Dataset):
    def __init__(self, pairs, input_vocab, output_vocab):
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_tensor = torch.tensor([self.input_vocab.get(c, self.input_vocab['<UNK>']) for c in src], dtype=torch.long)
        tgt_tensor = torch.tensor([self.output_vocab.get(c, self.output_vocab['<UNK>']) for c in tgt + Config.eos_token], dtype=torch.long)
        return src_tensor, tgt_tensor

# ----------------------- MODEL -----------------------
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        rnn_class = getattr(nn, cell_type)
        self.rnn = rnn_class(embedding_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        rnn_class = getattr(nn, cell_type)
        self.rnn = rnn_class(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden):
        if input.dim() == 0:
            input = input.unsqueeze(0)
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(Config.device)
        hidden = self.encoder(src)
        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

# ----------------------- UTILITIES -----------------------
def build_vocab(data):
    data = [str(d) for d in data if isinstance(d, str) or not pd.isna(d)]
    chars = set(c for word in data for c in word)
    vocab = {c: i+4 for i, c in enumerate(sorted(chars))}
    vocab['<PAD>'] = 0
    vocab['<SOS>'] = 1
    vocab['<EOS>'] = 2
    vocab['<UNK>'] = 3
    return vocab

def pad_tensor(tensor, length):
    return torch.cat([tensor, torch.zeros(length - len(tensor), dtype=torch.long)])

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) + 1 for t in tgt_batch]  # +1 for <SOS>
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    src_padded = torch.stack([pad_tensor(s, max_src_len) for s in src_batch])
    tgt_padded = torch.stack([pad_tensor(torch.cat((torch.tensor([1]), t)), max_tgt_len) for t in tgt_batch])

    return src_padded.to(Config.device), tgt_padded.to(Config.device)

# ----------------------- TRAINING -----------------------
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        optimizer.zero_grad()
        output = model(src, tgt, Config.teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ----------------------- MAIN -----------------------
if __name__ == '__main__':
    print("\n📂 Mounting Google Drive")
    drive.mount('/content/drive')

    print("\n📤 Upload the training TSV file (format: Latin -> Native)")
    uploaded_train = files.upload()
    train_file = list(uploaded_train.keys())[0]

    print("\n📤 Upload the testing TSV file (format: Latin -> Native)")
    uploaded_test = files.upload()
    test_file = list(uploaded_test.keys())[0]

    df_train = pd.read_csv(train_file, sep='\t', header=None)
    df_test = pd.read_csv(test_file, sep='\t', header=None)
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    train_pairs = [(str(src), str(tgt)) for src, tgt in zip(df_train[1], df_train[0]) if pd.notna(src) and pd.notna(tgt)]
    test_pairs = [(str(src), str(tgt)) for src, tgt in zip(df_test[1], df_test[0]) if pd.notna(src) and pd.notna(tgt)]

    src_data = [p[0] for p in train_pairs]
    tgt_data = [p[1] for p in train_pairs]

    input_vocab = build_vocab(src_data)
    output_vocab = build_vocab(tgt_data)
    inv_output_vocab = {v: k for k, v in output_vocab.items()}

    train_dataset = TransliterationDataset(train_pairs, input_vocab, output_vocab)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = Encoder(len(input_vocab), Config.embedding_dim, Config.hidden_dim, Config.num_layers, Config.cell_type).to(Config.device)
    decoder = Decoder(len(output_vocab), Config.embedding_dim, Config.hidden_dim, Config.num_layers, Config.cell_type).to(Config.device)
    model = Seq2Seq(encoder, decoder).to(Config.device)

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n🚀 Training Started")
    for epoch in range(Config.epochs):
        loss = train_model(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {loss:.4f}")

    def predict(word, max_len=30):
        model.eval()
        with torch.no_grad():
            src_indices = np.array([input_vocab.get(c, input_vocab['<UNK>']) for c in word], dtype=np.int64)
            src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(Config.device)
            tgt_tensor = torch.zeros((1, max_len), dtype=torch.long).to(Config.device)
            tgt_tensor[0, 0] = output_vocab['<SOS>']
            outputs = []
            hidden = model.encoder(src_tensor)
            input_token = tgt_tensor[0, 0]
            for t in range(1, max_len):
                output, hidden = model.decoder(input_token, hidden)
                pred_token = output.argmax(-1).item()
                if pred_token == output_vocab['<EOS>']:
                    break
                char = inv_output_vocab.get(pred_token, '<UNK>')
                outputs.append(char)
                input_token = torch.tensor(pred_token).to(Config.device)
            return ''.join(outputs)

    print("\n🔤 Sample Test Predictions:")
    for src, _ in test_pairs[:10]:
        print(f"{src} → {predict(str(src))}")

    print("\n✅ Char-Level Accuracy on Test Set:")
    y_true = [str(tgt) for _, tgt in test_pairs]
    y_pred = [predict(str(src)) for src, _ in test_pairs]

    def char_accuracy(y_true, y_pred):
        total_chars = sum(len(y) for y in y_true)
        correct_chars = sum(max(len(y) - distance(y, p), 0) for y, p in zip(y_true, y_pred))
        accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        return max(0.0, min(accuracy, 1.0))

    acc = char_accuracy(y_true, y_pred)
    print(f"Char-Level Accuracy on Test Set: {acc:.2%}")
