"""
src/aggregate.py

Aggregate per-seed evaluation JSONs across conditions → Table A and Table B.

Usage (after all 15 eval jobs finish):
    python -m src.aggregate \
        --results-dir checkpoints/../results \
        --split test \
        --out results/ablation_tables.json

Output JSON has two keys:
    table_A  — overall pixel F1/F2, mean ± std per condition
    table_B  — D2-only pixel F2 + bootstrap 95% CI + permutation p-value

Also prints both tables to stdout in a readable format.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CONDITION_NAMES = {
    1: "Baseline (BCE, random sample, no skip)",
    2: "+ biased sampling",
    3: "+ Focal+Tversky loss",
    4: "+ U-Net skip connections",
    5: "+ copy-paste (full system)",
}

TEST_SCENE = "Tromso_20241220"


def load_seed_result(json_path: Path, split: str) -> dict:
    """
    Load one per-seed evaluation JSON and flatten to the dict structure
    expected by format_ablation_tables.

    Keys returned:
        best_f1, best_f2, auprc           — from overall sweep
        dscale_f2                          — {int(d): float}
        bootstrap_d2                       — ci dict for D2 (d=2)
        perm_p_value                       — float (1.0 if val split)
    """
    with open(json_path) as f:
        data = json.load(f)

    overall = data["overall"]
    result: dict = {
        "best_f1": overall["best_f1"],
        "best_f2": overall["best_f2"],
        "auprc":   overall.get("auprc", float("nan")),
        "thr_f1":  overall.get("thr_f1", float("nan")),
        "thr_f2":  overall.get("thr_f2", float("nan")),
    }
    # Frozen-threshold metrics (deployment-mode), if present
    if "frozen" in overall:
        result["frozen"] = overall["frozen"]

    scene_r = data.get("scene_results", {}).get(TEST_SCENE, {})
    result["dscale_f2"]   = {int(d): float(v) for d, v in scene_r.get("dscale_f2", {}).items()}
    result["bootstrap_d2"] = scene_r.get("bootstrap_ci", {}).get("2", {})
    result["perm_p_value"] = scene_r.get("perm_d2", {}).get("p_value", 1.0)

    return result


def collect_condition_results(
    results_dir: Path,
    split: str,
    conditions: list[int] = [1, 2, 3, 4, 5],
    seeds: list[int]      = [0, 1, 2],
) -> dict[int, list[dict]]:
    """
    Scan results_dir for eval_cond{C}_seed{S}_{split}.json files.
    Returns {condition: [seed0_dict, seed1_dict, ...]}.
    Missing files are skipped with a warning.
    """
    condition_results: dict[int, list[dict]] = {}
    for cond in conditions:
        seed_dicts = []
        for seed in seeds:
            fname = results_dir / f"eval_cond{cond}_seed{seed}_{split}.json"
            if not fname.exists():
                log.warning("Missing: %s", fname)
                continue
            try:
                seed_dicts.append(load_seed_result(fname, split))
            except Exception as e:
                log.warning("Failed to load %s: %s", fname, e)
        if seed_dicts:
            condition_results[cond] = seed_dicts
        else:
            log.warning("No results for condition %d — skipped", cond)
    return condition_results


def print_table_a(rows: list[dict]) -> None:
    header = (
        f"{'#':<2}  {'Condition':<40}  "
        f"{'F1 mean ± std':<16}  {'F2 mean ± std':<16}  "
        f"{'thr_F2 mean ± std':<20}"
    )
    print("\n=== Table A: Overall Pixel F1 / F2 (sweep-mode) + Threshold Stability ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        f1s = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
        f2s = f"{r['f2_mean']:.4f} ± {r['f2_std']:.4f}"
        thr_mean = r.get('thr_f2_mean', float('nan'))
        thr_std  = r.get('thr_f2_std',  float('nan'))
        thrs = f"{thr_mean:.3f} ± {thr_std:.3f}"
        print(
            f"{r['condition']:<2}  {r['name']:<40}  "
            f"{f1s:<16}  {f2s:<16}  {thrs:<20}"
        )

    # Frozen-threshold sub-table (deployment-mode), if present
    if rows and "frozen" in rows[0]:
        thresholds = sorted(rows[0]["frozen"].keys())
        print(f"\n=== Table A2: Frozen-Threshold (Deployment-Mode) F2 ===")
        head = f"{'#':<2}  {'Condition':<40}"
        for t in thresholds:
            head += f"  {'F2@'+t:<16}"
        print(head)
        print("-" * len(head))
        for r in rows:
            line = f"{r['condition']:<2}  {r['name']:<40}"
            for t in thresholds:
                f = r.get("frozen", {}).get(t, {})
                fm = f.get("f2_mean", float("nan"))
                fs = f.get("f2_std",  float("nan"))
                line += f"  {fm:.4f} ± {fs:.4f}"
            print(line)


def print_table_b(rows: list[dict]) -> None:
    header = (
        f"{'#':<2}  {'Condition':<40}  "
        f"{'D2 F2':>8}  {'±':>6}  "
        f"{'95% CI':>14}  {'n':>4}  {'perm p':>8}"
    )
    print("\n=== Table B: D2-Only Pixel F2 (n=25, bootstrap 95% CI) ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['d2_ci_lower_mean']:.4f}, {r['d2_ci_upper_mean']:.4f}]"
        print(
            f"{r['condition']:<2}  {r['name']:<40}  "
            f"{r['d2_f2_mean']:>8.4f}  {r['d2_f2_std']:>6.4f}  "
            f"{ci:>14}  {r['n_d2']:>4}  {r['perm_p_value_mean']:>8.4f}"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stderr,
    )
    p = argparse.ArgumentParser(description="Aggregate eval JSONs → ablation tables")
    p.add_argument("--results-dir", required=True, type=Path,
                   help="Directory containing eval_cond*_seed*_*.json files")
    p.add_argument("--split",  default="test", choices=["val", "test"])
    p.add_argument("--out",    required=True,  type=Path,
                   help="Output JSON path for ablation tables")
    p.add_argument("--conditions", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--seeds",      nargs="+", type=int, default=[0,1,2])
    args = p.parse_args()

    from src.evaluate import format_ablation_tables

    condition_results = collect_condition_results(
        args.results_dir, args.split, args.conditions, args.seeds
    )
    if not condition_results:
        log.error("No results found in %s", args.results_dir)
        sys.exit(1)

    tables = format_ablation_tables(condition_results)

    print_table_a(tables["table_A"])
    print_table_b(tables["table_B"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tables, f, indent=2)
    log.info("Ablation tables saved → %s", args.out)


if __name__ == "__main__":
    main()
