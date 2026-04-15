"""
Parse seq_<jobid>.err log and produce a per-(condition, seed) overfit summary.

Usage:
    python scripts/analyze_overfit.py logs/seq_34643093.err

Output: a table with best AUPRC, epoch of best, train loss at peak vs exit,
post-warmup non-improvement count, and an overfit-slope metric.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# Match the per-run "Condition X | seed Y | device" line in train.py's stderr output
TRAIN_HDR = re.compile(r"Condition\s+(\d+)\s+\|\s+seed\s+(\d+)\s+\|\s+device")
EPOCH = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+loss=([\d.]+)\s+seg=[\d.]+\s+area=[\d.]+\s+\|\s+"
    r"val\s+F1=([\d.]+)\s+F2=([\d.]+)\s+AUPRC=([\d.]+)"
)
EARLY_STOP = re.compile(r"Early stopping at epoch (\d+)")

WARMUP_EPOCHS = 10


def parse(log_path: Path) -> list[dict]:
    text = log_path.read_text()
    runs = []
    cur = None
    for line in text.splitlines():
        m = TRAIN_HDR.search(line)
        if m:
            if cur is not None:
                runs.append(cur)
            cur = {
                "condition": int(m.group(1)),
                "seed":      int(m.group(2)),
                "epochs":    [],
                "early_stop_epoch": None,
            }
            continue
        if cur is None:
            continue
        m = EPOCH.search(line)
        if m:
            cur["epochs"].append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_f1":  float(m.group(3)),
                "val_f2":  float(m.group(4)),
                "auprc":   float(m.group(5)),
            })
            continue
        m = EARLY_STOP.search(line)
        if m:
            cur["early_stop_epoch"] = int(m.group(1))
    if cur is not None:
        runs.append(cur)
    return runs


def summarize(run: dict) -> dict:
    eps = run["epochs"]
    if not eps:
        return {"condition": run["condition"], "seed": run["seed"], "status": "no epochs"}

    # Best AUPRC
    best_idx = max(range(len(eps)), key=lambda i: eps[i]["auprc"])
    best = eps[best_idx]
    last = eps[-1]

    # Train loss change from peak to exit
    train_drop = best["train_loss"] - last["train_loss"]   # positive = train kept improving
    auprc_drop = best["auprc"] - last["auprc"]              # positive = val AUPRC declined
    overfit_slope = auprc_drop / max(train_drop, 1e-6)      # AUPRC lost per train-loss point

    # Post-warmup non-improvement count
    post_warmup = [e for e in eps if e["epoch"] > WARMUP_EPOCHS]
    if post_warmup:
        peak_after_warmup = max(e["auprc"] for e in post_warmup)
        no_improve = sum(1 for e in post_warmup if e["auprc"] <= peak_after_warmup * 0.999)
    else:
        no_improve = 0

    return {
        "cond":       run["condition"],
        "seed":       run["seed"],
        "epochs":     len(eps),
        "best_auprc": best["auprc"],
        "best_epoch": best["epoch"],
        "best_f2":    best["val_f2"],
        "exit_auprc": last["auprc"],
        "auprc_drop": auprc_drop,
        "train_drop": train_drop,
        "overfit_slope": overfit_slope,
        "early_stop": run["early_stop_epoch"],
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: analyze_overfit.py <seq_log.err>", file=sys.stderr)
        sys.exit(1)
    log_path = Path(sys.argv[1])
    runs = parse(log_path)
    if not runs:
        print(f"No training runs found in {log_path}")
        sys.exit(0)

    print(f"\nParsed {len(runs)} run(s) from {log_path.name}\n")
    print(f"{'cond':<5} {'seed':<5} {'epochs':<7} {'best_AUPRC':<11} {'@epoch':<7} "
          f"{'best_F2':<8} {'exit_AUPRC':<11} {'AUPRC_drop':<11} {'overfit_slope':<14} {'early_stop':<10}")
    print("-" * 110)
    for run in runs:
        s = summarize(run)
        if s.get("status") == "no epochs":
            print(f"{s['cond']:<5} {s['seed']:<5} (no epochs yet)")
            continue
        print(
            f"{s['cond']:<5} {s['seed']:<5} {s['epochs']:<7} "
            f"{s['best_auprc']:<11.4f} {s['best_epoch']:<7} "
            f"{s['best_f2']:<8.4f} {s['exit_auprc']:<11.4f} "
            f"{s['auprc_drop']:<11.4f} {s['overfit_slope']:<14.4f} "
            f"{str(s['early_stop'] or '-'):<10}"
        )


if __name__ == "__main__":
    main()
