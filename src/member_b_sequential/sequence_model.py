"""Member B - sequential model. A GRU or small Transformer over each user's
ordered early-window event stream, predicting activeness (is_inactive),
benchmarked against the tabular baseline on the shared split.

Each event carries: action bits (click/like/share/comment/view-comment/homepage),
feed position, dwell time, time gap since the previous event, and the CARD it was
(type image/video + primary content category, embedded). Order, timing, and what
was shown are the information the tabular baseline discards.

    python -m src.member_b_sequential.sequence_model --model gru --epochs 20 --n_seeds 5
    python -m src.member_b_sequential.sequence_model --features actions   # ablation: no content
    python -m src.member_b_sequential.sequence_model --features content   # ablation: no action bits
"""
import argparse, numpy as np, polars as pl
from src import config

DER = config.DERIVED_DIR
ACTIONS = ["isClick", "isComment", "isIntoPersonalHomepage", "isShare", "isViewComment", "isLike"]
NUM = ACTIONS + ["pos", "logview", "loggap"]     # numeric per-event features (action bits are cols 0..5)
N_ACTION = len(ACTIONS)
MAXLEN = 50
TOP_CATS = 200


def build_arrays(sample_users=None, seed=42):
    cache = DER / (f"seq_arrays_{sample_users}.npz" if sample_users else "seq_arrays.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if "users" in d.files:      # newer cache format includes user ids
            print(f"[loaded cached arrays from {cache.name}]")
            return d["Xn"], d["Xt"], d["Xc"], d["y"], d["sp"], d["users"], int(d["nnum"])
        print(f"[cache {cache.name} is old format; rebuilding]")
    print("building sequence arrays (one-off; cached for next run)...")
    ev = pl.read_parquet(DER / "early_events.parquet")
    lab = pl.read_parquet(DER / "user_modeling_table.parquet").select(["userId", "is_inactive", "split"])
    ev = ev.join(lab, on="userId", how="inner")
    if sample_users:
        keep = ev.select("userId").unique().sample(n=min(sample_users, ev["userId"].n_unique()), seed=seed)
        ev = ev.join(keep, on="userId", how="inner")
    ev = ev.sort(["userId", "impressTime", "impressPosition"])
    ev = ev.with_columns([
        (pl.col("impressPosition").cast(pl.Float64) / 10.0).alias("pos"),
        (pl.col("viewTime").fill_null(0.0).clip(0, None) + 1).log().alias("logview"),
        ((pl.col("impressTime") - pl.col("impressTime").shift(1).over("userId")).fill_null(0)
            .cast(pl.Float64).clip(0, None) / 1000.0 + 1).log().alias("loggap"),
        pl.col("card_type").fill_null(0).cast(pl.Int64).alias("type_idx"),
    ])
    tr_cats = (ev.filter(pl.col("split") == "train").drop_nulls("content_cat")
                 ["content_cat"].value_counts(sort=True).head(TOP_CATS)["content_cat"].to_list())
    vocab = {c: i + 1 for i, c in enumerate(tr_cats)}
    ev = ev.with_columns(pl.col("content_cat").replace_strict(vocab, default=0).alias("cat_idx"))
    ev = ev.with_columns(pl.int_range(0, pl.len()).over("userId").alias("step")).filter(pl.col("step") < MAXLEN)
    udf = ev.group_by("userId", maintain_order=True).agg(
        pl.col("is_inactive").first(), pl.col("split").first())
    uix = {u: i for i, u in enumerate(udf["userId"].to_list())}
    ev = ev.with_columns(pl.col("userId").replace_strict(uix).alias("ui"))
    n = udf.height
    ui = ev["ui"].to_numpy(); st = ev["step"].to_numpy()
    Xn = np.zeros((n, MAXLEN, len(NUM)), np.float32); Xn[ui, st, :] = ev.select(NUM).to_numpy()
    Xt = np.zeros((n, MAXLEN), np.int64);           Xt[ui, st] = ev["type_idx"].to_numpy()
    Xc = np.zeros((n, MAXLEN), np.int64);           Xc[ui, st] = ev["cat_idx"].to_numpy()
    y = udf["is_inactive"].to_numpy().astype(np.int64)
    sp = udf["split"].to_numpy()
    users = np.array(udf["userId"].to_list())
    np.savez(cache, Xn=Xn, Xt=Xt, Xc=Xc, y=y, sp=sp, users=users, nnum=len(NUM))
    return Xn, Xt, Xc, y, sp, users, len(NUM)


def fit_predict(Xn, Xt, Xc, y, sp, kind="gru", epochs=20, hidden=48, seed=0,
                inc_actions=True, inc_content=True, verbose=False):
    """Train once and return test-set (y, predicted prob) in array order."""
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nnum = Xn.shape[2]

    class SeqClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.inc_content = inc_content
            d = nnum
            if inc_content:
                self.type_emb = nn.Embedding(3, 2)
                self.cat_emb = nn.Embedding(TOP_CATS + 1, 8)
                d += 10
            self.kind = kind
            if kind == "gru":
                self.enc = nn.GRU(d, hidden, batch_first=True)
            else:
                self.proj = nn.Linear(d, hidden)
                layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden,
                                                   batch_first=True, dropout=0.1)
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, xn, xt, xc):
            x = xn if not self.inc_content else torch.cat([xn, self.type_emb(xt), self.cat_emb(xc)], -1)
            mask = (xt == 0) & (xc == 0) & (xn.abs().sum(-1) == 0)
            if self.kind == "gru":
                _, hn = self.enc(x); h = hn[-1]
            else:
                z = self.enc(self.proj(x), src_key_padding_mask=mask)
                z = z.masked_fill(mask.unsqueeze(-1), 0.0)
                h = z.sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
            return self.fc(h).squeeze(-1)

    def tens(m):
        xn = Xn[m].copy()
        if not inc_actions:
            xn[:, :, :N_ACTION] = 0.0     # ablation: drop action bits
        return (torch.tensor(xn), torch.tensor(Xt[m]), torch.tensor(Xc[m]),
                torch.tensor(y[m], dtype=torch.float32))
    tr = [t.to(dev) for t in tens(sp == "train")]
    Xn_e, Xt_e, Xc_e, y_e = tens(sp == "test")
    model = SeqClf().to(dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - tr[3].mean()) / tr[3].mean()]).to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 2048
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr[3]))
        for i in range(0, len(tr[3]), bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = lossf(model(tr[0][idx], tr[1][idx], tr[2][idx]), tr[3][idx]); loss.backward(); opt.step()
        if verbose:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xn_e.to(dev), Xt_e.to(dev), Xc_e.to(dev))).cpu().numpy()
            print(f"  epoch {ep+1}: ROC-AUC={roc_auc_score(y_e, p):.4f}  PR-AUC={average_precision_score(y_e, p):.4f}")
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xn_e.to(dev), Xt_e.to(dev), Xc_e.to(dev))).cpu().numpy()
    return y_e.numpy().astype(int), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gru", "transformer"], default="gru")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--sample_users", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--features", choices=["all", "actions", "content"], default="all",
                    help="ablation: all | actions-only (no content) | content-only (no action bits)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score, average_precision_score

    Xn, Xt, Xc, y, sp, users, nnum = build_arrays(args.sample_users or None)
    inc_actions = args.features in ("all", "actions")
    inc_content = args.features in ("all", "content")
    print(f"users={len(y)}  inactive rate={y.mean():.3f}  model={args.model}  features={args.features}  n_seeds={args.n_seeds}")

    aucs, prs = [], []
    for s in range(args.n_seeds):
        y_e, p = fit_predict(Xn, Xt, Xc, y, sp, args.model, args.epochs, args.hidden, s,
                             inc_actions, inc_content, args.verbose)
        a, pr = roc_auc_score(y_e, p), average_precision_score(y_e, p)
        aucs.append(a); prs.append(pr)
        print(f"seed {s}: ROC-AUC={a:.4f}  PR-AUC={pr:.4f}")
    aucs, prs = np.array(aucs), np.array(prs)
    if args.n_seeds > 1:
        print(f"\n=== {args.model} ({args.features}) over {args.n_seeds} seeds ===")
        print(f"ROC-AUC: mean {aucs.mean():.4f}  std {aucs.std():.4f}  range [{aucs.min():.4f}, {aucs.max():.4f}]")
        print(f"PR-AUC : mean {prs.mean():.4f}  std {prs.std():.4f}  range [{prs.min():.4f}, {prs.max():.4f}]")
    print("Baseline (tabular, early features): ROC-AUC 0.705 / PR-AUC 0.832")


if __name__ == "__main__":
    main()
