import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import random
import string
import math
import os


# CONFIG

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 128
DROPOUT = 0.1

BATCH_SIZE = 32
GROUP_SIZE = 6    
RL_UPDATES = 1500      
LR = 1e-5               
MAX_LEN = 20

TEMPERATURE = 0.9
ENTROPY_COEF = 0.01
KL_COEF = 0.02
GRAD_CLIP = 1.0

EVAL_EVERY = 100
SAVE_PATH = "minigpt_reverse_skip_b_step2_rl.pth"
STEP1_MODEL_PATH = "minigpt_reverse_step1.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# VOCABULARY

special_tokens = ["<PAD>", "<BOS>", "<SEP>", "<EOS>"]
letters = list(string.ascii_lowercase)

itos = special_tokens + letters
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_ID = stoi["<PAD>"]
BOS_ID = stoi["<BOS>"]
SEP_ID = stoi["<SEP>"]
EOS_ID = stoi["<EOS>"]
B_ID = stoi["b"]

VOCAB_SIZE = len(itos)



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


# DATA GENERATION

def generate_input_string(min_len=2, max_len=6):
    length = random.randint(min_len, max_len)
    return "".join(random.choice(letters) for _ in range(length))


def make_prompt_ids(text):
    tokens = ["<BOS>"] + list(text) + ["<SEP>"]
    return [stoi[t] for t in tokens]


def make_target_skip_b(text):
    
    reversed_chars = list(reversed(text))
    filtered = [ch for ch in reversed_chars if ch != "b"]
    return "".join(filtered)


def decode_generated_ids(ids):
    
    chars = []

    for token_id in ids:
        if token_id == EOS_ID:
            break

        token = itos[token_id]

        if token in ["<PAD>", "<BOS>", "<SEP>", "<EOS>"]:
            continue

        chars.append(token)

    return "".join(chars)


# REWARD FUNCTION

def lcs_length(a, b):
    
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[n][m]


def compute_reward(output, target, ended_with_eos=True):
   

    reward = 0.0

    if output == target:
        reward += 8.0

    max_len = max(len(output), len(target), 1)
    lcs = lcs_length(output, target)
    similarity = lcs / max_len
    reward += 4.0 * similarity

    correct_pos = 0
    for o, t in zip(output, target):
        if o == t:
            correct_pos += 1

    pos_acc = correct_pos / max(len(target), 1)
    reward += 2.0 * pos_acc

    b_count = output.count("b")
    if b_count == 0:
        reward += 2.0
    else:
        reward -= 4.0 * b_count

    length_diff = abs(len(output) - len(target))
    reward -= 0.8 * length_diff

    if ended_with_eos:
        reward += 1.0
    else:
        reward -= 1.0

    if target == "":
        if output == "":
            reward += 4.0
        else:
            reward -= 2.0 * len(output)

    return reward


# GENERATION FOR TRAINING

def sample_response_with_logprobs(model, ref_model, text):
    

    model.train()
    ref_model.eval()

    prompt_ids = make_prompt_ids(text)
    ids = prompt_ids[:]

    generated_ids = []
    log_probs = []
    ref_log_probs = []
    entropies = []

    target = make_target_skip_b(text)
    max_new_tokens = max(len(target) + 3, 3)

    ended_with_eos = False

    for _ in range(max_new_tokens):
        if len(ids) >= MAX_LEN:
            break

        input_ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        logits = model(input_ids)
        next_logits = logits[0, -1] / TEMPERATURE

        next_logits[PAD_ID] = -1e9
        next_logits[BOS_ID] = -1e9
        next_logits[SEP_ID] = -1e9

        probs = F.softmax(next_logits, dim=-1)
        dist = Categorical(probs=probs)

        next_id = dist.sample()
        log_prob = dist.log_prob(next_id)
        entropy = dist.entropy()

        with torch.no_grad():
            ref_logits = ref_model(input_ids)
            ref_next_logits = ref_logits[0, -1] / TEMPERATURE

            ref_next_logits[PAD_ID] = -1e9
            ref_next_logits[BOS_ID] = -1e9
            ref_next_logits[SEP_ID] = -1e9

            ref_log_prob_all = F.log_softmax(ref_next_logits, dim=-1)
            ref_log_prob = ref_log_prob_all[next_id]

        token_id = next_id.item()

        ids.append(token_id)
        generated_ids.append(token_id)
        log_probs.append(log_prob)
        ref_log_probs.append(ref_log_prob)
        entropies.append(entropy)

        if token_id == EOS_ID:
            ended_with_eos = True
            break

    output = decode_generated_ids(generated_ids)
    reward = compute_reward(output, target, ended_with_eos=ended_with_eos)

    if len(log_probs) == 0:
        # Safety fallback
        log_probs = [torch.tensor(0.0, device=DEVICE, requires_grad=True)]
        ref_log_probs = [torch.tensor(0.0, device=DEVICE)]
        entropies = [torch.tensor(0.0, device=DEVICE)]

    return {
        "text": text,
        "target": target,
        "output": output,
        "generated_ids": generated_ids,
        "reward": reward,
        "log_probs": torch.stack(log_probs),
        "ref_log_probs": torch.stack(ref_log_probs),
        "entropies": torch.stack(entropies),
    }


# GREEDY GENERATION FOR EVALUATION

@torch.no_grad()
def generate_greedy(model, text, max_new_tokens=10):
    model.eval()

    ids = make_prompt_ids(text)

    for _ in range(max_new_tokens):
        if len(ids) >= MAX_LEN:
            break

        input_ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = model(input_ids)
        next_logits = logits[0, -1]

        next_logits[PAD_ID] = -1e9
        next_logits[BOS_ID] = -1e9
        next_logits[SEP_ID] = -1e9

        next_id = torch.argmax(next_logits).item()
        ids.append(next_id)

        if next_id == EOS_ID:
            break

    prompt_len = len(make_prompt_ids(text))
    generated_ids = ids[prompt_len:]
    output = decode_generated_ids(generated_ids)

    full_tokens = [itos[i] for i in ids]

    return output, full_tokens


@torch.no_grad()
def evaluate_model(model, n_samples=500):
    model.eval()

    exact = 0
    no_b = 0
    total = 0
    char_score_sum = 0.0

    for _ in range(n_samples):
        text = generate_input_string()
        target = make_target_skip_b(text)

        output, _ = generate_greedy(
            model,
            text,
            max_new_tokens=max(len(target) + 3, 3)
        )

        if output == target:
            exact += 1

        if "b" not in output:
            no_b += 1

        lcs = lcs_length(output, target)
        char_score = lcs / max(len(output), len(target), 1)
        char_score_sum += char_score

        total += 1

    exact_acc = exact / total
    no_b_rate = no_b / total
    char_acc = char_score_sum / total

    return {
        "exact_acc": exact_acc,
        "no_b_rate": no_b_rate,
        "char_acc": char_acc,
    }


@torch.no_grad()
def test_examples(model):
    examples = [
        "hello",
        "abc",
        "abcde",
        "bob",
        "banana",
        "bbbb",
        "dog",
        "apple",
        "bottle",
        "cab"
    ]

    print("\nRL Skip-b Test Examples")

    for ex in examples:
        target = make_target_skip_b(ex)
        output, tokens = generate_greedy(
            model,
            ex,
            max_new_tokens=max(len(target) + 3, 3)
        )

        print(f"Input:    {ex}")
        print(f"Expected: {target}")
        print(f"Output:   {output}")
        print(f"Correct:  {output == target}")
        print(f"Tokens:   {tokens}")
        print()


# RL TRAINING: GRPO-STYLE

def train_rl():
    if not os.path.exists(STEP1_MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find Step 1 model: {STEP1_MODEL_PATH}\n"
            f"Please run Step 1 training first to create {STEP1_MODEL_PATH}"
        )

    model = MiniGPT().to(DEVICE)
    model.load_state_dict(torch.load(STEP1_MODEL_PATH, map_location=DEVICE))

    ref_model = MiniGPT().to(DEVICE)
    ref_model.load_state_dict(torch.load(STEP1_MODEL_PATH, map_location=DEVICE))
    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"Device: {DEVICE}")

    best_exact = 0.0

    for update in range(1, RL_UPDATES + 1):
        batch_texts = [generate_input_string() for _ in range(BATCH_SIZE)]

        all_losses = []
        all_rewards = []
        all_outputs = []

        for text in batch_texts:
            group_samples = []

            for _ in range(GROUP_SIZE):
                sample = sample_response_with_logprobs(model, ref_model, text)
                group_samples.append(sample)

            rewards = torch.tensor(
                [s["reward"] for s in group_samples],
                dtype=torch.float32,
                device=DEVICE
            )

            mean_reward = rewards.mean()
            std_reward = rewards.std(unbiased=False)

            if std_reward.item() < 1e-6:
                advantages = rewards - mean_reward
            else:
                advantages = (rewards - mean_reward) / (std_reward + 1e-8)

            for sample, adv in zip(group_samples, advantages):
                log_probs = sample["log_probs"]
                ref_log_probs = sample["ref_log_probs"]
                entropies = sample["entropies"]

                policy_log_prob = log_probs.mean()

                approx_kl = (log_probs - ref_log_probs).mean()

                entropy_bonus = entropies.mean()

                pg_loss = -adv.detach() * policy_log_prob

                loss = pg_loss + KL_COEF * approx_kl - ENTROPY_COEF * entropy_bonus

                all_losses.append(loss)
                all_rewards.append(sample["reward"])
                all_outputs.append(sample["output"])

        if len(all_losses) == 0:
            continue

        total_loss = torch.stack(all_losses).mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        avg_reward = sum(all_rewards) / len(all_rewards)

        if update % 20 == 0:
            print(
                f"Update {update:04d} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Avg Reward: {avg_reward:.3f}"
            )

        if update % EVAL_EVERY == 0:
            metrics = evaluate_model(model, n_samples=500)

            print(
                f"\n[Eval @ Update {update:04d}] "
                f"Exact Acc: {metrics['exact_acc']:.3f} | "
                f"Char Acc: {metrics['char_acc']:.3f} | "
                f"No-b Rate: {metrics['no_b_rate']:.3f}"
            )

            test_examples(model)

            if metrics["exact_acc"] > best_exact:
                best_exact = metrics["exact_acc"]
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"New best model saved to {SAVE_PATH}")

    # Final save
    torch.save(model.state_dict(), SAVE_PATH)
    print("\nModel saved")

    final_metrics = evaluate_model(model, n_samples=1000)
    print("\nFinal Evaluation")
    print(f"Exact Match Accuracy: {final_metrics['exact_acc']:.4f}")
    print(f"Character Accuracy:   {final_metrics['char_acc']:.4f}")
    print(f"No-b Rate:            {final_metrics['no_b_rate']:.4f}")

    test_examples(model)


# MAIN

if __name__ == "__main__":
    train_rl()