"""How a masked language model scores a word in context.

Idea: take a sentence like "People from Bihar are lazy."
Hide ("mask") each piece of the word "lazy", one piece at a time, and ask the model how
probable the hidden piece was. Add up the log-probabilities. A higher (less negative) score
means the model finds "lazy" more natural in that sentence.
"""
import numpy as np


def build_sentence(template, identity, attribute):
    """Fill the template and return (sentence, (start, end)) where (start, end) is the exact
    character span of the attribute word inside the sentence."""
    pre, post = template.split("{attribute}")
    pre = pre.replace("{identity}", identity)
    post = post.replace("{identity}", identity)
    return pre + attribute + post, (len(pre), len(pre) + len(attribute))


def build_masked_variants(tokenizer, sentence, span, max_length=64):
    """Returns (original_ids, positions, variants): one variant per token that belongs to the attribute."""
    enc = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=max_length)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    s, e = span
    positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
    if not positions:
        raise ValueError(f"Attribute tokens not found in: {sentence!r}")
    variants = []
    for p in positions:
        v = list(ids)
        v[p] = tokenizer.mask_token_id
        variants.append(v)
    return ids, positions, variants


def score_sentences(tokenizer, logprob_fn, items, batch_size=64, max_length=64, progress=None):
    """items: list of (sentence, span). Returns a list of scores (sum of log-probs over attribute tokens)."""
    flat_ids, flat_pos, flat_tgt, owner = [], [], [], []
    for k, (sent, span) in enumerate(items):
        ids, positions, variants = build_masked_variants(tokenizer, sent, span, max_length)
        for p, v in zip(positions, variants):
            flat_ids.append(v)
            flat_pos.append(p)
            flat_tgt.append(ids[p])
            owner.append(k)
    out = np.zeros(len(items), dtype=np.float64)
    n = len(flat_ids)
    rng = range(0, n, batch_size)
    if progress:
        rng = progress(rng)
    for i in rng:
        lp = logprob_fn(flat_ids[i:i + batch_size], flat_pos[i:i + batch_size], flat_tgt[i:i + batch_size])
        for j, val in enumerate(lp):
            out[owner[i + j]] += float(val)
    return out.tolist()


def load_masked_lm(model_name):
    """Loads a Hugging Face masked language model. Returns (tokenizer, logprob_fn)."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(f"{model_name}: needs a fast tokenizer (offset mapping). See Troubleshooting.")
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    @torch.no_grad()
    def logprob_fn(batch_ids, positions, targets):
        width = max(len(x) for x in batch_ids)
        ids = torch.full((len(batch_ids), width), pad, dtype=torch.long)
        att = torch.zeros((len(batch_ids), width), dtype=torch.long)
        for r, x in enumerate(batch_ids):
            ids[r, :len(x)] = torch.tensor(x)
            att[r, :len(x)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        rows = torch.arange(len(batch_ids), device=device)
        at_pos = logits[rows, torch.tensor(positions, device=device)]          # (batch, vocab)
        logp = torch.log_softmax(at_pos.float(), dim=-1)
        picked = logp[rows, torch.tensor(targets, device=device)]
        return picked.cpu().tolist()

    return tok, logprob_fn
