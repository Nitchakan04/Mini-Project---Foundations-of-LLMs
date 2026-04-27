import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import string

# Config
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 128
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
MAX_LEN = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Vocabulary
special_tokens = ["<PAD>", "<BOS>", "<SEP>", "<EOS>"]
letters = list(string.ascii_lowercase)

itos = special_tokens + letters
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_ID = stoi["<PAD>"]
BOS_ID = stoi["<BOS>"]
SEP_ID = stoi["<SEP>"]
EOS_ID = stoi["<EOS>"]

VOCAB_SIZE = len(itos)


# Dataset
def generate_sample():
    length = random.randint(2, 6)
    x = [random.choice(letters) for _ in range(length)]
    y = list(reversed(x))

    sequence = ["<BOS>"] + x + ["<SEP>"] + y + ["<EOS>"]
    ids = [stoi[t] for t in sequence]

    return ids, "".join(x), "".join(y)


class ReverseDataset(Dataset):
    def __init__(self, n_samples):
        self.samples = [generate_sample() for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, x, y = self.samples[idx]

        ids = ids[:MAX_LEN]
        padding = [PAD_ID] * (MAX_LEN - len(ids))
        ids = ids + padding

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        return input_ids, target_ids


# Mini GPT Model
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=N_LAYERS
        )

        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()

        x = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(x)

        return logits


# Evaluation
@torch.no_grad()
def generate(model, text, max_new_tokens=10):
    model.eval()

    tokens = ["<BOS>"] + list(text) + ["<SEP>"]
    ids = [stoi[t] for t in tokens]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        if input_ids.shape[1] >= MAX_LEN:
            break

        logits = model(input_ids)
        next_id = torch.argmax(logits[0, -1]).item()

        ids.append(next_id)

        if next_id == EOS_ID:
            break

    decoded = [itos[i] for i in ids]
    return decoded


@torch.no_grad()
def test_examples(model):
    examples = ["hello", "abc", "train", "dog", "apple"]

    print("\nTest Examples")
    for ex in examples:
        output = generate(model, ex)
        print(f"Input: {ex}")
        print("Output tokens:", output)
        print()


# Train
def train():
    train_dataset = ReverseDataset(20000)
    val_dataset = ReverseDataset(2000)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MiniGPT().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            logits = model(input_ids)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(DEVICE)
                target_ids = target_ids.to(DEVICE)

                logits = model(input_ids)

                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    target_ids.reshape(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if epoch % 5 == 0:
            test_examples(model)

    torch.save(model.state_dict(), "minigpt_reverse_step1.pth")
    print("\nModel saved")


if __name__ == "__main__":
    train()