"""Step 2: turn raw scores into findings.
Run:  python -m src.analyze            (uses results/scores.csv)
      python -m src.analyze --small    (uses results/scores_small.csv)

What it computes, in plain words
1. For each sentence pair (identity, attribute), the model gives a score.
2. For each attribute, we subtract the average over all identities. This removes the fact that
   some words are simply more common than others, so only differences BETWEEN identities remain.
3. An identity's "negative association" = how much more (than average) it is linked to negative
   attributes minus how much it is linked to positive attributes. Above 0 = linked to negative
   words more than the average identity; below 0 = linked to positive words more.
4. We resample the attributes 1000 times to get a 95% range, and check if English and Hindi agree.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src import config


def association(sub):
    """sub: rows for one model, one language, (all templates averaged). Returns Series identity -> score."""
    piv = sub.groupby(["identity", "attribute", "polarity"])["score"].mean().reset_index()
    wide = piv.pivot(index="identity", columns="attribute", values="score")
    z = wide.sub(wide.mean(axis=0), axis=1)
    pol = piv.drop_duplicates("attribute").set_index("attribute")["polarity"]
    neg = [a for a in z.columns if pol[a] == "neg"]
    pos = [a for a in z.columns if pol[a] == "pos"]
    return z, neg, pos, (z[neg].mean(axis=1) - z[pos].mean(axis=1))


def bootstrap(z, neg, pos, n=config.N_BOOTSTRAP, seed=config.SEED):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        nn = rng.choice(neg, size=len(neg), replace=True)
        pp = rng.choice(pos, size=len(pos), replace=True)
        draws.append((z[nn].mean(axis=1) - z[pp].mean(axis=1)).values)
    d = np.array(draws)
    return np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    path = os.path.join(config.RESULTS_DIR, "scores_small.csv" if args.small else "scores.csv")
    df = pd.read_csv(path)

    rows, assoc_store, tmpl_store = [], {}, {}
    for (model, lang), sub in df.groupby(["model", "lang"]):
        z, neg, pos, a = association(sub)
        lo, hi = bootstrap(z, neg, pos)
        assoc_store[(model, lang)] = a
        for i, ident in enumerate(a.index):
            rows.append({"model": model, "lang": lang, "identity": ident, "neg_association": round(a[ident], 3),
                         "ci_low": round(lo[i], 3), "ci_high": round(hi[i], 3)})
        for tid, tsub in sub.groupby("template_id"):                       # one template at a time
            tmpl_store[(model, lang, tid)] = association(tsub)[3]
    assoc = pd.DataFrame(rows)
    assoc.to_csv(os.path.join(config.RESULTS_DIR, "association.csv"), index=False)

    corr = []
    def add(kind, model, a, b, la, lb):
        common = a.index.intersection(b.index)
        if len(common) >= 4:
            r = spearmanr(a[common], b[common])
            corr.append({"check": kind, "model": model, "pair": f"{la} vs {lb}", "n": len(common),
                         "spearman": round(float(r.statistic), 3), "p_value": round(float(r.pvalue), 3)})
    models = sorted(df.model.unique())
    for m in models:
        if (m, "en") in assoc_store and (m, "hi") in assoc_store:
            add("English vs Hindi (same model)", m, assoc_store[(m, "en")], assoc_store[(m, "hi")], "en", "hi")
        for lang in ("en", "hi"):
            ts = sorted({t for (mm, ll, t) in tmpl_store if mm == m and ll == lang})
            if len(ts) >= 2:
                add(f"Template robustness ({lang})", m, tmpl_store[(m, lang, ts[0])], tmpl_store[(m, lang, ts[1])], ts[0], ts[1])
    for lang in ("en", "hi"):
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                if (m1, lang) in assoc_store and (m2, lang) in assoc_store:
                    add(f"Model agreement ({lang})", f"{m1}/{m2}", assoc_store[(m1, lang)], assoc_store[(m2, lang)], m1, m2)
    pd.DataFrame(corr).to_csv(os.path.join(config.RESULTS_DIR, "correlations.csv"), index=False)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    langs = sorted(assoc.lang.unique())
    fig, axes = plt.subplots(len(models), len(langs), figsize=(5 * len(langs), 3.2 * len(models)), squeeze=False)
    for r, m in enumerate(models):
        for c, lg in enumerate(langs):
            s = assoc[(assoc.model == m) & (assoc.lang == lg)].sort_values("neg_association")
            ax = axes[r][c]
            ax.errorbar(s.neg_association, range(len(s)), xerr=[s.neg_association - s.ci_low, s.ci_high - s.neg_association],
                        fmt="o", capsize=2)
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(s.identity, fontsize=7)
            ax.set_title(f"{m} ({lg})", fontsize=9)
            if r == len(models) - 1:
                ax.set_xlabel("negative association (95% range)", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "association.png"), dpi=150)

    print("\n=== Association (top and bottom per model/language) ===")
    for (m, lg), a in assoc_store.items():
        a = a.sort_values()
        print(f"{m} {lg}: lowest = {list(a.index[:2])}, highest = {list(a.index[-2:])}")
    print("\n=== Checks ===")
    print(pd.DataFrame(corr).to_string(index=False) if corr else "not enough data")
    print(f"\nSaved association.csv, correlations.csv, association.png in {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
