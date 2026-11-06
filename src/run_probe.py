"""Step 1: score every sentence with every model.
Run:  python -m src.run_probe --small          (quick test, 1 to 2 minutes)
      python -m src.run_probe                  (full run)
      python -m src.run_probe --models xlmr    (one model only)
"""
import argparse
import os
import pandas as pd
from tqdm import tqdm
from src import config
from src.scoring import build_sentence, load_masked_lm, score_sentences


def build_table(small=False):
    ids = pd.read_csv(os.path.join(config.DATA_DIR, "identities.csv"))
    attrs = pd.read_csv(os.path.join(config.DATA_DIR, "attributes.csv"))
    temps = pd.read_csv(os.path.join(config.DATA_DIR, "templates.csv"))
    if small:
        ids = ids.head(3)
        attrs = pd.concat([attrs[attrs.polarity == "pos"].head(2), attrs[attrs.polarity == "neg"].head(2)])
    rows = []
    for _, t in temps.iterrows():
        for _, i in ids.iterrows():
            for _, a in attrs.iterrows():
                ident = i[f"identity_{t.lang}"]
                attr = a[f"attr_{t.lang}"]
                sent, span = build_sentence(t.template, ident, attr)
                rows.append({"lang": t.lang, "template_id": t.template_id, "identity": i["identity_en"],
                             "attribute": a["attr_en"], "polarity": a["polarity"], "sentence": sent,
                             "span_start": span[0], "span_end": span[1]})
    return pd.DataFrame(rows)


def check_tokenizer(tok, attrs_hi):
    """Warn if Hindi words turn into [UNK] (that would make results meaningless)."""
    bad = [w for w in attrs_hi if tok.unk_token_id is not None and tok.unk_token_id in tok(w, add_special_tokens=False)["input_ids"]]
    sample = attrs_hi[0]
    print(f"[check] Hindi sample {sample!r} -> {tok.tokenize(sample)}")
    if bad:
        print(f"[WARNING] These Hindi words contain [UNK] tokens for this model: {bad}. See Troubleshooting.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.DEFAULT_MODELS)
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()

    table = build_table(args.small)
    print(f"[data] {len(table)} sentences per model")
    print(table.sample(4, random_state=0)[["lang", "sentence"]].to_string(index=False))
    attrs_hi = pd.read_csv(os.path.join(config.DATA_DIR, "attributes.csv"))["attr_hi"].tolist()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_scores = []
    for key in args.models:
        tok, fn = load_masked_lm(config.MODELS[key])
        check_tokenizer(tok, attrs_hi)
        items = [(r.sentence, (r.span_start, r.span_end)) for r in table.itertuples()]
        scores = score_sentences(tok, fn, items, batch_size=config.BATCH_SIZE, max_length=config.MAX_LENGTH,
                                 progress=lambda r: tqdm(r, desc=key))
        df = table.copy()
        df["model"] = key
        df["score"] = scores
        all_scores.append(df)
        print(f"[done] {key}")
    out = pd.concat(all_scores, ignore_index=True)
    name = "scores_small.csv" if args.small else "scores.csv"
    out.drop(columns=["span_start", "span_end"]).to_csv(os.path.join(config.RESULTS_DIR, name), index=False)
    print(f"Saved {config.RESULTS_DIR}/{name}")


if __name__ == "__main__":
    main()
