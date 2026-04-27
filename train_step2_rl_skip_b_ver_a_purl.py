import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import string
import os


# CONFIG
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 128
DROPOUT = 0.1
MAX_LEN = 20

# Pure GRPO settings
BATCH_SIZE = 16
GROUP_SIZE = 8
RL_UPDATES = 500
LR = 2e-5

TEMPERATURE = 1.3
ENTROPY_COEF = 0.02
KL_COEF = 0.003
GRAD_CLIP = 1.0

PRINT_EVERY = 1
EVAL_EVERY = 25

STEP1_MODEL_PATH = "minigpt_reverse_step1.pth"
SAVE_PATH = "minigpt_reverse_step2_pure_rl.pth"

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


# DATA FUNCTIONS

def generate_input_string(min_len=2, max_len=6):
    """
    Generate random input string.
    เพิ่มโอกาสให้มี b เพื่อให้ RL เจอ skip-b cases บ่อยขึ้น
    """
    length = random.randint(min_len, max_len)

    if random.random() < 0.75:
        chars = [random.choice(letters) for _ in range(length)]
        n_b = random.randint(1, min(2, length))
        positions = random.sample(range(length), n_b)

        for pos in positions:
            chars[pos] = "b"

        return "".join(chars)

    return "".join(random.choice(letters) for _ in range(length))


def make_target_skip_b(text):
   
    return "".join([ch for ch in reversed(text) if ch != "b"])


def make_prompt_ids(text):
    tokens = ["<BOS>"] + list(text) + ["<SEP>"]
    return [stoi[t] for t in tokens]


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


# LCS FOR PARTIAL CREDIT

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


# ACTION MASKING

def apply_action_mask(logits, target_len, generated_len):

    logits = logits.clone()

    if generated_len >= target_len:
        forced = torch.full_like(logits, -1e9)
        forced[EOS_ID] = 0.0
        return forced

    logits[PAD_ID] = -1e9
    logits[BOS_ID] = -1e9
    logits[SEP_ID] = -1e9

    logits[B_ID] = -1e9

    logits[EOS_ID] = -1e9

    return logits


# REWARD FUNCTION

def compute_reward(output, target, ended_with_eos=True):
  

    reward = 0.0

    if output == target:
        reward += 100.0
    else:
        reward -= 10.0

    if len(output) == len(target):
        reward += 30.0
    else:
        reward -= 20.0 * abs(len(output) - len(target))

    min_len = min(len(output), len(target))
    for i in range(min_len):
        if output[i] == target[i]:
            reward += 10.0
        else:
            reward -= 5.0

    max_len = max(len(output), len(target), 1)
    lcs = lcs_length(output, target)
    reward += 20.0 * (lcs / max_len)

    b_count = output.count("b")
    reward -= 40.0 * b_count

    if b_count == 0:
        reward += 5.0

    if target != "" and output == "":
        reward -= 40.0

    if len(output) < len(target):
        reward -= 15.0 * (len(target) - len(output))

    if len(output) > len(target):
        reward -= 15.0 * (len(output) - len(target))

    if ended_with_eos:
        reward += 5.0
    else:
        reward -= 10.0

    if target == "":
        if output == "":
            reward += 100.0
        else:
            reward -= 40.0 * len(output)

    return reward


# SAMPLING FOR GRPO

def sample_response_with_logprobs(model, ref_model, text):
    model.train()
    ref_model.eval()

    target = make_target_skip_b(text)
    target_len = len(target)

    prompt_ids = make_prompt_ids(text)
    ids = prompt_ids[:]

    generated_ids = []
    log_probs = []
    ref_log_probs = []
    entropies = []

    max_new_tokens = target_len + 1

    ended_with_eos = False

    for _ in range(max_new_tokens):
        if len(ids) >= MAX_LEN:
            break

        generated_len = len(ids) - len(prompt_ids)

        input_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        logits = model(input_ids)
        next_logits = logits[0, -1] / TEMPERATURE

        next_logits = apply_action_mask(
            next_logits,
            target_len=target_len,
            generated_len=generated_len
        )

        probs = F.softmax(next_logits, dim=-1)
        dist = Categorical(probs=probs)

        next_id = dist.sample()
        log_prob = dist.log_prob(next_id)
        entropy = dist.entropy()

        with torch.no_grad():
            ref_logits = ref_model(input_ids)
            ref_next_logits = ref_logits[0, -1] / TEMPERATURE

            ref_next_logits = apply_action_mask(
                ref_next_logits,
                target_len=target_len,
                generated_len=generated_len
            )

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
    reward = compute_reward(output, target, ended_with_eos)

    if len(log_probs) == 0:
        log_probs = [torch.tensor(0.0, device=DEVICE, requires_grad=True)]
        ref_log_probs = [torch.tensor(0.0, device=DEVICE)]
        entropies = [torch.tensor(0.0, device=DEVICE)]

    return {
        "text": text,
        "target": target,
        "output": output,
        "reward": reward,
        "log_probs": torch.stack(log_probs),
        "ref_log_probs": torch.stack(ref_log_probs),
        "entropies": torch.stack(entropies),
    }


# GREEDY GENERATION FOR EVALUATION

@torch.no_grad()
def generate_greedy(model, text):
    model.eval()

    target = make_target_skip_b(text)
    target_len = len(target)

    prompt_ids = make_prompt_ids(text)
    ids = prompt_ids[:]

    max_new_tokens = target_len + 1

    for _ in range(max_new_tokens):
        if len(ids) >= MAX_LEN:
            break

        generated_len = len(ids) - len(prompt_ids)

        input_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        logits = model(input_ids)
        next_logits = logits[0, -1]

        next_logits = apply_action_mask(
            next_logits,
            target_len=target_len,
            generated_len=generated_len
        )

        next_id = torch.argmax(next_logits).item()
        ids.append(next_id)

        if next_id == EOS_ID:
            break

    generated_ids = ids[len(prompt_ids):]
    output = decode_generated_ids(generated_ids)
    full_tokens = [itos[i] for i in ids]

    return output, full_tokens


# EVALUATION

@torch.no_grad()
def evaluate_model(model, n_samples=500):
    model.eval()

    fixed_examples = [
        "hello",
        "abc",
        "abcde",
        "bob",
        "banana",
        "bbbb",
        "dog",
        "apple",
        "bottle",
        "cab",
    ]

    eval_samples = fixed_examples + [
        generate_input_string() for _ in range(n_samples)
    ]

    exact = 0
    no_b = 0
    char_score_sum = 0.0

    for text in eval_samples:
        target = make_target_skip_b(text)
        output, _ = generate_greedy(model, text)

        if output == target:
            exact += 1

        if "b" not in output:
            no_b += 1

        lcs = lcs_length(output, target)
        char_score = lcs / max(len(output), len(target), 1)
        char_score_sum += char_score

    total = len(eval_samples)

    return {
        "exact_acc": exact / total,
        "char_acc": char_score_sum / total,
        "no_b_rate": no_b / total,
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
        "cab",
    ]

    print("\nImproved Pure GRPO Skip-b Test Examples")

    for ex in examples:
        target = make_target_skip_b(ex)
        output, tokens = generate_greedy(model, ex)

        print(f"Input:    {ex}")
        print(f"Expected: {target}")
        print(f"Output:   {output}")
        print(f"Correct:  {output == target}")
        print(f"Tokens:   {tokens}")
        print()


# PURE GRPO TRAINING

def train_pure_grpo():
    if not os.path.exists(STEP1_MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find Step 1 model: {STEP1_MODEL_PATH}\n"
            f"Please run Step 1 first to create {STEP1_MODEL_PATH}"
        )

    print(f"Device: {DEVICE}", flush=True)

    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: CUDA not available. CPU training will be slow.", flush=True)

    # Policy model starts directly from Step 1
    model = MiniGPT().to(DEVICE)
    model.load_state_dict(torch.load(STEP1_MODEL_PATH, map_location=DEVICE))

    # Frozen reference model is also Step 1 model
    ref_model = MiniGPT().to(DEVICE)
    ref_model.load_state_dict(torch.load(STEP1_MODEL_PATH, map_location=DEVICE))
    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"Loaded Step 1 model from: {STEP1_MODEL_PATH}", flush=True)
    print("Start Improved Pure GRPO-style RL...", flush=True)

    print("\nBefore RL Evaluation")
    metrics = evaluate_model(model, n_samples=300)

    print(
        f"Exact Acc: {metrics['exact_acc']:.3f} | "
        f"Char Acc: {metrics['char_acc']:.3f} | "
        f"No-b Rate: {metrics['no_b_rate']:.3f}",
        flush=True
    )

    test_examples(model)

    best_exact = metrics["exact_acc"]

    for update in range(1, RL_UPDATES + 1):
        batch_texts = [generate_input_string() for _ in range(BATCH_SIZE)]

        all_losses = []
        all_rewards = []

        for text in batch_texts:
            group_samples = []

            # GRPO group sampling
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

            # Group-relative advantage
            if std_reward.item() < 1e-6:

                advantages = rewards - mean_reward

                if torch.allclose(advantages, torch.zeros_like(advantages)):
                    advantages = rewards / (rewards.abs().mean() + 1e-8)
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

                loss = (
                    pg_loss
                    + KL_COEF * approx_kl
                    - ENTROPY_COEF * entropy_bonus
                )

                all_losses.append(loss)
                all_rewards.append(sample["reward"])

        total_loss = torch.stack(all_losses).mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        avg_reward = sum(all_rewards) / len(all_rewards)

        if update % PRINT_EVERY == 0:
            print(
                f"Update {update:04d} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Avg Reward: {avg_reward:.3f}",
                flush=True
            )

        if update % EVAL_EVERY == 0:
            metrics = evaluate_model(model, n_samples=500)

            print(
                f"\n[Eval Update {update:04d}] "
                f"Exact Acc: {metrics['exact_acc']:.3f} | "
                f"Char Acc: {metrics['char_acc']:.3f} | "
                f"No-b Rate: {metrics['no_b_rate']:.3f}",
                flush=True
            )

            test_examples(model)

            if metrics["exact_acc"] > best_exact:
                best_exact = metrics["exact_acc"]
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"New model saved", flush=True)

    torch.save(model.state_dict(), SAVE_PATH)

    print("\nImproved Pure GRPO model saved", flush=True)

    final_metrics = evaluate_model(model, n_samples=1000)

    print("\nFinal Evaluation")
    print(f"Exact Match Accuracy: {final_metrics['exact_acc']:.4f}")
    print(f"Character Accuracy:   {final_metrics['char_acc']:.4f}")
    print(f"No-b Rate:            {final_metrics['no_b_rate']:.4f}")

    test_examples(model)


# MAIN

if __name__ == "__main__":
    train_pure_grpo()