#!/usr/bin/env python3
"""
Graph-based AML / fraud-ring detection  (runs fully locally, CPU / CUDA / Apple-MPS)

Graph      : accounts = nodes, transfers = directed edges
Task       : classify each TRANSACTION (edge) as laundering / not (edge classification)
Model      : directed GraphSAGE (separate in/out neighbourhood aggregation, 2 hops)
             + MLP edge head on [src_emb, dst_emb, edge_features]
Baseline   : gradient boosting on the same tabular features, NO graph message passing
Evaluation : strict temporal split (60/20/20). Node features and the message-passing
             graph for each split only contain edges that existed by that time
             -> no look-ahead leakage.
Post-hoc   : flagged transactions -> connected "ring" candidates (size, cycle flag, $)

Data       : IBM "Transactions for Anti Money Laundering" (HI-Small_Trans.csv etc.)
             https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
             Use --demo to generate a synthetic file with planted rings to smoke-test.

Only dependency beyond numpy/pandas/scipy/sklearn/networkx/matplotlib is PyTorch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

# --------------------------------------------------------------------------- #
# 1. Data loading  (edit COLMAP if your CSV uses different column names)
# --------------------------------------------------------------------------- #
COLMAP = {  # IBM AML header: the two "Account" columns become Account / Account.1
    "Timestamp": "ts", "From Bank": "src_bank", "Account": "src_acct",
    "To Bank": "dst_bank", "Account.1": "dst_acct",
    "Amount Received": "amount_received", "Receiving Currency": "recv_cur",
    "Amount Paid": "amount_paid", "Payment Currency": "pay_cur",
    "Payment Format": "payment_format", "Is Laundering": "label",
}


def load_csv(path, max_rows):
    df = pd.read_csv(path, nrows=max_rows if max_rows else None)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLMAP)
    need = set(COLMAP.values())
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns {missing}. Adjust COLMAP in aml_gnn.py.")
    df["ts"] = pd.to_datetime(df["ts"])
    df["t"] = (df["ts"] - df["ts"].min()).dt.total_seconds()
    df["hour"] = df["ts"].dt.hour
    return df


# --------------------------------------------------------------------------- #
# 2. Synthetic demo data with planted laundering typologies
# --------------------------------------------------------------------------- #
def make_demo(path, n_acct=6000, n_tx=250_000, n_rings=120, seed=0):
    rng = np.random.default_rng(seed)
    days = 15
    fmts = ["Reinvestment", "Cheque", "Credit Card", "ACH", "Cash", "Wire", "Bitcoin"]
    cur = ["US Dollar", "Euro", "Yuan", "Bitcoin"]
    bank = rng.integers(0, 40, n_acct)
    rows = []

    def add(s, d, t, amt, lab, fmt=None):
        rows.append((t, bank[s], s, bank[d], d, amt, fmt or rng.choice(fmts), lab))

    # background: heavy-tailed amounts, sticky counterparties (hard negatives)
    fav = {a: rng.integers(0, n_acct, 4) for a in range(n_acct)}
    for _ in range(n_tx):
        s = rng.integers(0, n_acct)
        d = fav[s][rng.integers(0, 4)] if rng.random() < 0.5 else rng.integers(0, n_acct)
        if s != d:
            add(s, d, rng.uniform(0, days * 86400), float(np.exp(rng.normal(6, 1.6))), 0)
    # planted typologies
    shells = rng.choice(n_acct, n_acct // 6, replace=False)  # thinly-used shell accounts
    for _ in range(n_rings):
        kind = rng.choice(["cycle", "scatter_gather", "chain"])
        t0 = rng.uniform(0, days * 86400 - 86400)
        amt = float(np.exp(rng.normal(7.5, 1.0)))
        k = int(rng.integers(3, 7))
        accts = rng.choice(shells, k + 2, replace=False)
        if kind == "cycle":
            for i in range(k):
                add(accts[i], accts[(i + 1) % k], t0 + i * rng.uniform(600, 7200),
                    amt * (0.97 ** i), 1)
        elif kind == "scatter_gather":
            src, sink, mules = accts[0], accts[1], accts[2:]
            for i, m in enumerate(mules):
                add(src, m, t0 + rng.uniform(0, 3600), amt / len(mules), 1)
                add(m, sink, t0 + 3600 + rng.uniform(0, 7200), amt / len(mules) * 0.98, 1)
        else:
            for i in range(k + 1):
                add(accts[i], accts[i + 1], t0 + i * rng.uniform(300, 3600), amt * (0.95 ** i), 1)
    df = pd.DataFrame(rows, columns=["t", "fb", "a1", "tb", "a2", "amt", "fmt", "lab"])
    df["ts"] = pd.Timestamp("2022/09/01") + pd.to_timedelta(df["t"], unit="s")
    out = pd.DataFrame({
        "Timestamp": df["ts"].dt.strftime("%Y/%m/%d %H:%M"), "From Bank": df["fb"],
        "Account": df["a1"].map(lambda x: f"{x:08X}"), "To Bank": df["tb"],
        "Account.1": df["a2"].map(lambda x: f"{x:08X}"), "Amount Received": df["amt"].round(2),
        "Receiving Currency": rng.choice(cur, len(df)), "Amount Paid": df["amt"].round(2),
        "Payment Currency": rng.choice(cur, len(df)), "Payment Format": df["fmt"],
        "Is Laundering": df["lab"],
    })
    out.to_csv(path, index=False)
    print(f"[demo] wrote {len(out):,} tx, {int(out['Is Laundering'].sum()):,} laundering -> {path}")


# --------------------------------------------------------------------------- #
# 3. Graph construction + features
# --------------------------------------------------------------------------- #
def prepare(df):
    """Sort by time, integer-encode accounts, build causal per-edge features."""
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    n = len(df)
    sk = df.src_bank.astype(str) + "_" + df.src_acct.astype(str)
    dk = df.dst_bank.astype(str) + "_" + df.dst_acct.astype(str)
    codes, uniq = pd.factorize(pd.concat([sk, dk], ignore_index=True))
    src, dst, N = codes[:n].astype(np.int64), codes[n:].astype(np.int64), len(uniq)

    amt = df.amount_paid.to_numpy(float)
    la = np.log1p(np.abs(amt))
    g = pd.Series(la).groupby(src)
    cn = g.cumcount().to_numpy()
    prev_mean = np.where(cn > 0, (g.cumsum().to_numpy() - la) / np.maximum(cn, 1), la)
    gap = pd.Series(df.t.to_numpy()).groupby(src).diff().fillna(0).to_numpy()

    key = src * N + dst
    first = pd.Series(np.arange(n)).groupby(key).min()
    fi = first.reindex(dst * N + src).to_numpy()              # first time reverse pair appeared
    rev_earlier = (fi < np.arange(n)).astype(float)           # NaN compares False

    # layering signature: how soon / how much does this sender move money out after last receiving funds
    tt = df.t.to_numpy()
    left = pd.DataFrame({"node": src, "t": tt, "i": np.arange(n)})
    right = pd.DataFrame({"node": dst, "t": tt, "t_recv": tt, "ramt": np.abs(amt)})
    mg = pd.merge_asof(left, right, on="t", by="node", direction="backward",
                       allow_exact_matches=False).sort_values("i")
    has_in = mg.ramt.notna().to_numpy()
    since_recv = np.where(has_in, np.log1p(np.nan_to_num(tt - mg.t_recv.to_numpy())), 0.0)
    ratio_recv = np.where(has_in, la - np.log1p(mg.ramt.fillna(0).to_numpy()), 0.0)

    fmt = pd.get_dummies(df.payment_format).astype(float).to_numpy()
    h = df.hour.to_numpy() * 2 * np.pi / 24
    ef = np.column_stack([
        la, la - prev_mean,
        (df.src_bank != df.dst_bank).to_numpy(float),
        (df.pay_cur != df.recv_cur).to_numpy(float),
        (src == dst).astype(float), rev_earlier, (cn == 0).astype(float),
        np.log1p(gap), np.sin(h), np.cos(h),
        has_in.astype(float), since_recv, ratio_recv,
        np.abs(ratio_recv), np.exp(-10 * np.abs(ratio_recv)) * has_in,      # amount-conserving relay
        (has_in & (since_recv < np.log1p(6 * 3600))).astype(float),         # forwarded within 6h
        fmt,
    ]).astype(np.float32)
    return dict(df=df, src=src, dst=dst, N=N, amt=amt, ef=ef,
                y=df.label.to_numpy(np.float32), t=df.t.to_numpy(), n=n)


def node_features(N, src, dst, amt, use_cycles):
    """Structural / behavioural account profile from the edges seen so far."""
    a = np.abs(amt)
    od, idg = np.bincount(src, minlength=N), np.bincount(dst, minlength=N)
    oa = np.bincount(src, weights=a, minlength=N)
    ia = np.bincount(dst, weights=a, minlength=N)
    pairs = np.unique(src * N + dst)
    ps, pdst = pairs // N, pairs % N
    uo, ui = np.bincount(ps, minlength=N), np.bincount(pdst, minlength=N)
    pass_through = np.minimum(oa, ia) / (np.maximum(oa, ia) + 1)  # ~1 => relay / mule
    fan = (od - idg) / (od + idg + 1)                              # +1 fan-out, -1 fan-in
    has_rev = np.isin(pdst * N + ps, pairs)
    recip = np.bincount(ps[has_rev], minlength=N)
    if use_cycles:  # directed 3-cycles through each node: diag(A^3)
        A = sp.csr_matrix((np.ones(len(ps), np.float32), (ps, pdst)), shape=(N, N))
        A.setdiag(0); A.eliminate_zeros()
        tri = np.asarray((A @ A).multiply(A.T).sum(1)).ravel()
    else:
        tri = np.zeros(N)
    x = np.column_stack([np.log1p(od), np.log1p(idg), np.log1p(oa), np.log1p(ia),
                         np.log1p(uo), np.log1p(ui), pass_through, fan,
                         np.log1p(recip), np.log1p(tri)]).astype(np.float32)
    return x


class Snapshot:
    """Message-passing graph from edges in the trailing time window ending at edge m.
    A fixed window keeps node features stationary across train/val/test."""

    def __init__(self, P, m, dev, use_cycles, window_s):
        lo = int(np.searchsorted(P["t"], P["t"][m - 1] - window_s)) if window_s else 0
        s, d = P["src"][lo:m], P["dst"][lo:m]
        self.x_np = node_features(P["N"], s, d, P["amt"][lo:m], use_cycles)
        self.dev = dev
        self.src = torch.from_numpy(s).to(dev)
        self.dst = torch.from_numpy(d).to(dev)
        N = P["N"]
        self.inv_in = torch.from_numpy(1.0 / np.maximum(np.bincount(d, minlength=N), 1)).float().to(dev)
        self.inv_out = torch.from_numpy(1.0 / np.maximum(np.bincount(s, minlength=N), 1)).float().to(dev)
        self.x = None

    def set_scaling(self, mu, sd):
        self.x = torch.from_numpy((self.x_np - mu) / sd).float().to(self.dev)


# --------------------------------------------------------------------------- #
# 4. Model
# --------------------------------------------------------------------------- #
class DirSAGE(nn.Module):
    """h' = W_s h + W_i mean(in-neighbours) + W_o mean(out-neighbours)."""

    def __init__(self, i, o):
        super().__init__()
        self.ws, self.wi, self.wo = nn.Linear(i, o), nn.Linear(i, o, bias=False), nn.Linear(i, o, bias=False)

    def forward(self, h, g):
        a_in = torch.zeros_like(h).index_add_(0, g.dst, h[g.src]) * g.inv_in[:, None]
        a_out = torch.zeros_like(h).index_add_(0, g.src, h[g.dst]) * g.inv_out[:, None]
        return self.ws(h) + self.wi(a_in) + self.wo(a_out)


class EdgeGNN(nn.Module):
    def __init__(self, nf, ef, hid=64, drop=0.3):
        super().__init__()
        self.l1, self.l2 = DirSAGE(nf, hid), DirSAGE(hid, hid)
        self.n1, self.n2 = nn.LayerNorm(hid), nn.LayerNorm(hid)
        self.enc = nn.Sequential(nn.Linear(ef, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(3 * hid, hid), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(hid, 1))
        self.skip = nn.Linear(ef, 1)  # wide path: raw edge features go straight to the logit
        self.drop = drop

    def embed(self, g):
        h = F.dropout(F.relu(self.n1(self.l1(g.x, g))), self.drop, self.training)
        return F.relu(self.n2(self.l2(h, g))) + h  # residual keeps 1-hop signal

    def edge_logits(self, h, s, d, e):
        return (self.head(torch.cat([h[s], h[d], self.enc(e)], 1)) + self.skip(e)).squeeze(-1)


# --------------------------------------------------------------------------- #
# 5. Train / evaluate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score(model, g, P_src, P_dst, ef_t, idx, chunk=400_000):
    model.eval()
    h = model.embed(g)
    out = []
    for i in range(0, len(idx), chunk):
        j = torch.from_numpy(idx[i:i + chunk]).to(h.device)
        out.append(torch.sigmoid(model.edge_logits(h, P_src[j], P_dst[j], ef_t[j])).cpu())
    return torch.cat(out).numpy()


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(th[np.argmax(f1[:-1])]) if len(th) else 0.5


def metrics(y, p, thr):
    yhat = p >= thr
    tp, fp, fn = int((yhat & (y == 1)).sum()), int((yhat & (y == 0)).sum()), int((~yhat & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    k = int(y.sum())
    top = np.argsort(-p)[:k] if k else []
    return dict(pr_auc=float(average_precision_score(y, p)) if k else float("nan"),
                roc_auc=float(roc_auc_score(y, p)) if 0 < k < len(y) else float("nan"),
                threshold=thr, precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12),
                precision_at_k=float(y[top].mean()) if k else float("nan"),
                n_pos=k, n=len(y), flagged=int(yhat.sum()))


def run_baseline(P, snaps, ef, splits, seed):
    """Tabular GBM: edge features + endpoint profile features. No message passing."""
    (a, b, c) = splits

    def build(idx, snap):
        return np.hstack([ef[idx], snap.x_np[P["src"][idx]], snap.x_np[P["dst"][idx]]])

    tr = np.arange(0, a)
    rng = np.random.default_rng(seed)
    pos, neg = tr[P["y"][tr] == 1], tr[P["y"][tr] == 0]
    neg = rng.choice(neg, min(len(neg), 30 * max(len(pos), 1)), replace=False)
    sel = np.concatenate([pos, neg])
    w = np.where(P["y"][sel] == 1, len(neg) / max(len(pos), 1) / 5, 1.0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)
    clf.fit(build(sel, snaps["train"]), P["y"][sel], sample_weight=w)
    va, te = np.arange(a, b), np.arange(b, c)
    pv = clf.predict_proba(build(va, snaps["val"]))[:, 1]
    pt = clf.predict_proba(build(te, snaps["test"]))[:, 1]
    thr = best_threshold(P["y"][va], pv)
    return metrics(P["y"][te], pt, thr)


def find_rings(P, te_idx, p, thr, out_dir, top_n=30):
    import networkx as nx
    sel = te_idx[p >= thr]
    if len(sel) == 0:
        return pd.DataFrame()
    G = nx.DiGraph()
    df = P["df"]
    for i, sc in zip(sel, p[p >= thr]):
        s, d = int(P["src"][i]), int(P["dst"][i])
        w = abs(float(P["amt"][i]))
        if G.has_edge(s, d):
            G[s][d]["amount"] += w
            G[s][d]["score"] = max(G[s][d]["score"], float(sc))
            G[s][d]["fraud"] = max(G[s][d]["fraud"], int(P["y"][i]))
        else:
            G.add_edge(s, d, amount=w, score=float(sc), fraud=int(P["y"][i]))
    rows = []
    for k, comp in enumerate(nx.weakly_connected_components(G)):
        H = G.subgraph(comp)
        if H.number_of_edges() < 2:
            continue
        rows.append(dict(ring_id=k, n_accounts=H.number_of_nodes(), n_edges=H.number_of_edges(),
                         total_amount=sum(d["amount"] for _, _, d in H.edges(data=True)),
                         max_score=max(d["score"] for _, _, d in H.edges(data=True)),
                         has_cycle=not nx.is_directed_acyclic_graph(H),
                         confirmed_laundering_edges=sum(d["fraud"] for _, _, d in H.edges(data=True)),
                         nodes=sorted(comp)))
    rings = pd.DataFrame(rows)
    if rings.empty:
        return rings
    rings = rings.sort_values(["max_score", "n_edges"], ascending=False)
    rings.drop(columns="nodes").to_csv(out_dir / "ring_candidates.csv", index=False)
    try:  # picture of the top rings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keep = set().union(*[set(n) for n in rings.head(6)["nodes"]])
        H = G.subgraph(keep)
        pos = nx.spring_layout(H, seed=1, k=0.6)
        col = ["crimson" if d["fraud"] else "steelblue" for _, _, d in H.edges(data=True)]
        plt.figure(figsize=(11, 8))
        nx.draw_networkx(H, pos, node_size=60, with_labels=False, edge_color=col, arrows=True,
                         arrowsize=10, width=1.4, node_color="#333")
        plt.title("Top flagged sub-networks (red = confirmed laundering label, blue = no label)")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(out_dir / "flagged_rings.png", dpi=150); plt.close()
    except Exception as e:  # plotting is optional
        print("[warn] plot skipped:", e)
    return rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="IBM AML CSV (e.g. HI-Small_Trans.csv)")
    ap.add_argument("--demo", action="store_true", help="generate + use synthetic data")
    ap.add_argument("--max-rows", type=int, default=2_000_000, help="0 = all rows (needs lots of RAM)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--neg-ratio", type=int, default=20, help="negatives sampled per positive per step")
    ap.add_argument("--window-days", type=float, default=5, help="trailing window for graph/features; 0 = all history")
    ap.add_argument("--no-cycles", action="store_true", help="skip 3-cycle feature (saves RAM/time)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--out", default="runs/aml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                       else "cpu")
    print("device:", dev)

    if args.demo:
        args.data = str(out / "demo_trans.csv")
        make_demo(args.data)
    if not args.data:
        ap.error("provide --data <csv> or --demo")

    t0 = time.time()
    P = prepare(load_csv(args.data, args.max_rows))
    n = P["n"]
    a, b = int(n * 0.6), int(n * 0.8)
    print(f"{n:,} transactions | {P['N']:,} accounts | laundering rate {P['y'].mean():.4%} "
          f"| prep {time.time()-t0:.0f}s")
    for nm, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n))):
        print(f"  {nm}: {sl.stop - sl.start:,} tx, {int(P['y'][sl].sum()):,} positive")

    use_cyc = not args.no_cycles
    W = args.window_days * 86400
    snaps = {"train": Snapshot(P, a, dev, use_cyc, W), "val": Snapshot(P, b, dev, use_cyc, W),
             "test": Snapshot(P, n, dev, use_cyc, W)}
    mu = snaps["train"].x_np.mean(0); sd = snaps["train"].x_np.std(0) + 1e-6
    for s in snaps.values():
        s.set_scaling(mu, sd)
    emu, esd = P["ef"][:a].mean(0), P["ef"][:a].std(0) + 1e-6
    ef = (P["ef"] - emu) / esd
    ef_t = torch.from_numpy(ef).float().to(dev)
    S, D = torch.from_numpy(P["src"]).to(dev), torch.from_numpy(P["dst"]).to(dev)
    y = P["y"]

    model = EdgeGNN(snaps["train"].x.shape[1], ef.shape[1], args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    pos = np.where(y[:a] == 1)[0]; neg = np.where(y[:a] == 0)[0]
    if len(pos) == 0:
        raise SystemExit("No positive labels in the training window - use more rows.")
    va_idx, te_idx = np.arange(a, b), np.arange(b, n)
    best, best_state, patience = -1, None, 0
    g_tr = snaps["train"]

    for ep in range(1, args.epochs + 1):
        model.train()
        sel = np.concatenate([pos, np.random.choice(neg, min(len(neg), args.neg_ratio * len(pos)), replace=False)])
        j = torch.from_numpy(sel).to(dev)
        h = model.embed(g_tr)
        logit = model.edge_logits(h, S[j], D[j], ef_t[j])
        loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[sel]).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == args.epochs:
            pv = score(model, snaps["val"], S, D, ef_t, va_idx)
            ap_val = average_precision_score(y[va_idx], pv) if y[va_idx].sum() else 0.0
            print(f"epoch {ep:3d} loss {loss.item():.4f}  val PR-AUC {ap_val:.4f}")
            if ap_val > best:
                best, patience = ap_val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    print("early stop"); break
    model.load_state_dict(best_state)

    pv = score(model, snaps["val"], S, D, ef_t, va_idx)
    thr = best_threshold(y[va_idx], pv)
    pt = score(model, snaps["test"], S, D, ef_t, te_idx)
    res = {"gnn": metrics(y[te_idx], pt, thr)}
    if not args.no_baseline:
        res["baseline_gbm_no_graph"] = run_baseline(P, snaps, P["ef"], (a, b, n), args.seed)
    res["dataset"] = dict(transactions=n, accounts=P["N"], laundering_rate=float(y.mean()))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print("\n=== TEST RESULTS (temporal hold-out) ===")
    for k in ("gnn", "baseline_gbm_no_graph"):
        if k in res:
            m = res[k]
            print(f"{k:24s} PR-AUC {m['pr_auc']:.3f} | ROC-AUC {m['roc_auc']:.3f} | "
                  f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} | P@k {m['precision_at_k']:.3f}")

    torch.save({"state": model.state_dict(), "thr": thr, "mu": mu, "sd": sd, "emu": emu, "esd": esd,
                "args": vars(args)}, out / "model.pt")

    df = P["df"].iloc[te_idx].copy()
    df["gnn_score"] = pt
    df["flagged"] = pt >= thr
    df.sort_values("gnn_score", ascending=False).head(5000)[
        ["ts", "src_bank", "src_acct", "dst_bank", "dst_acct", "amount_paid", "pay_cur",
         "payment_format", "label", "gnn_score", "flagged"]].to_csv(out / "flagged_transactions.csv", index=False)
    rings = find_rings(P, te_idx, pt, thr, out)
    if len(rings):
        print(f"\n{len(rings)} ring candidates; top 5:")
        print(rings.drop(columns="nodes").head(5).to_string(index=False))
    print(f"\nOutputs in {out.resolve()}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
