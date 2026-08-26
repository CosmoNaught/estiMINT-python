"""Validate the learned posterior against the prior it was trained on.

Two modes, both operating on a held-out split:

``--prior`` (default)
    Histograms of the target column, showing the marginal (prior) distribution the
    simulator was sampled from: raw, log10, and standardized-log10.

``--sbc``
    Simulation-based calibration. Draws (theta, x) from the joint are already what a
    held-out split is, so the rank of the true theta under p(theta | x) must be uniform
    if the posterior is calibrated. Because the model is a monotone 1-D conditional
    flow with a standard-normal base, the infinite-sample rank statistic is available
    in closed form as the probability integral transform, PIT = P(Y <= y_true | x), so
    no posterior sampling is needed. Also reports the data-averaged posterior (DAP):
    pooling one posterior draw per x must reproduce the prior.

Examples:
    python -m estimint.v2.validate_posterior --target eir --splits test
    python -m estimint.v2.validate_posterior --sbc --splits test val
"""

import argparse
import logging
import textwrap
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy import stats

from .data.features import get_features
from .data.preprocess import PreparedData, prepare_data
from .models.rqs import ConditionalRQS, RQSArtifact

log = logging.getLogger(__name__)

SPLIT_ATTRS = {
    "train": "train_data",
    "val": "val_data",
    "calib": "calib_data",
    "test": "test_data",
}
SPLIT_NAMES = sorted(SPLIT_ATTRS)
BLUE, RED, GREY = "#4C72B0", "#C44E52", "#888888"

UNIFORM_MEAN = 0.5
UNIFORM_VAR = 1 / 12

Records = list[dict[str, np.ndarray]]


# --------------------------------------------------------------------------- data


def _split_records(prepared: PreparedData, split: str) -> Records:
    return getattr(prepared, SPLIT_ATTRS[split])


def _column(records: Records, key: str) -> np.ndarray:
    return np.array([rec[key] for rec in records], dtype=np.float64)


def _split_targets(records: Records) -> dict[str, np.ndarray]:
    """Raw, log10, and standardized-log10 targets for one split."""
    return {
        "raw": _column(records, "y_raw"),
        "log10": _column(records, "y"),
        "standardized": _column(records, "y_std"),
    }


def _sibling_groups(records: Records) -> list[np.ndarray]:
    """Row indices grouped by ``parameter_index``.

    Every parameter set is simulated several times and the target is a property of the
    parameter set, so sibling rows share the same theta. Their PIT values are correlated
    and a KS test over all rows would be anti-conservative, so SBC statistics are
    reported on subsets holding one row per parameter.
    """
    by_param: dict[int, list[int]] = {}
    for i, rec in enumerate(records):
        by_param.setdefault(int(rec["ps"][0]), []).append(i)
    return [np.array(idx, dtype=int) for idx in by_param.values()]


def _one_sim_per_parameter(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One row index per ``parameter_index``, chosen uniformly at random."""
    return np.array([g[rng.integers(len(g))] for g in groups], dtype=int)


def _describe(values: np.ndarray) -> str:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return (
        f"n={len(values)}  mean={values.mean():.3g}  sd={values.std():.3g}\n"
        f"min={values.min():.3g}  q1={q1:.3g}  median={med:.3g}  q3={q3:.3g}  max={values.max():.3g}"
    )


# ------------------------------------------------------------------------ prior


def plot_prior(
    split: str, targets: dict[str, np.ndarray], target_name: str, bins: int, path_png: Path
) -> None:
    """Draw raw / log10 / standardized histograms of one split's target values."""
    panels = [
        ("raw", f"{target_name} (raw)", True),
        ("log10", f"log10({target_name})", False),
        ("standardized", f"standardized log10({target_name})", False),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    for ax, (key, xlabel, log_x) in zip(axes, panels):
        values = targets[key]
        if log_x:
            # The raw target spans orders of magnitude, so bin it on a log axis.
            values = values[values > 0]
            bin_edges = np.logspace(np.log10(values.min()), np.log10(values.max()), bins + 1)
            ax.set_xscale("log")
        else:
            bin_edges = bins
        ax.hist(values, bins=bin_edges, color=BLUE, edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(_describe(values), fontsize=8, loc="left")

    n_sims = len(targets["raw"])
    fig.suptitle(f"{split} split — prior distribution of {target_name} ({n_sims} sims)")
    fig.tight_layout()
    _save(fig, path_png)


# -------------------------------------------------------------------------- SBC


def _uniformity_stats(pit: np.ndarray) -> dict[str, float]:
    ks = stats.kstest(pit, "uniform")
    return {
        "n": len(pit),
        "ks_stat": float(ks.statistic),
        "ks_pval": float(ks.pvalue),
        "mean": float(pit.mean()),  # 0.5 if unbiased; > 0.5 means predictions run low
        "var": float(pit.var()),  # 1/12 = 0.0833 if calibrated; larger = over-confident
    }


def _repeated_uniformity(
    pit: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator, repeats: int
) -> tuple[dict[str, float], np.ndarray]:
    """Aggregate the uniformity test over many independent subsamples.

    Which sibling represents a parameter set is arbitrary, and with only a few hundred
    parameters a single draw can flip the verdict. Repeating the subsample and reporting
    the median removes that coin flip; ``reject_frac`` shows how stable the call is.
    Returns the summary and the subsample whose KS p-value is closest to the median,
    which is what gets plotted.
    """
    draws = [_one_sim_per_parameter(groups, rng) for _ in range(repeats)]
    per_draw = [_uniformity_stats(pit[idx]) for idx in draws]
    pvals = np.array([d["ks_pval"] for d in per_draw])

    summary: dict[str, float] = {
        key: float(np.median([d[key] for d in per_draw]))
        for key in ("ks_stat", "ks_pval", "mean", "var")
    }
    summary["n"] = len(groups)
    summary["repeats"] = repeats
    summary["reject_frac"] = float((pvals < 0.05).mean())

    representative = draws[int(np.argmin(np.abs(pvals - np.median(pvals))))]
    return summary, representative


def _interpretation(summary: dict[str, float]) -> str:
    """One-line read of the rank-histogram shape from its first two moments."""
    mean, var, pval = summary["mean"], summary["var"], summary["ks_pval"]
    reject_frac = summary["reject_frac"]
    if pval >= 0.05 and reject_frac <= 0.25:
        return "calibrated (uniform PIT not rejected)"

    notes = []
    if pval >= 0.05:
        notes.append("borderline (verdict flips across subsamples)")
    if var > UNIFORM_VAR:
        notes.append("U-shaped: intervals too narrow (over-confident)")
    elif var < UNIFORM_VAR:
        notes.append("dome-shaped: intervals too wide (under-confident)")
    if mean > 0.53:
        notes.append("shifted: predictions biased low")
    elif mean < 0.47:
        notes.append("shifted: predictions biased high")
    return "; ".join(notes) or "non-uniform"


def _plot_pit_hist(ax, pit: np.ndarray, bins: int, summary: dict[str, float]) -> None:
    n = len(pit)
    ax.hist(pit, bins=bins, range=(0, 1), color=BLUE, edgecolor="white")
    expected = n / bins
    # Pointwise 95% binomial band for a single bin under the uniform null.
    half = 1.96 * np.sqrt(n * (1 / bins) * (1 - 1 / bins))
    ax.axhline(expected, color=RED, lw=1.2, label="uniform")
    ax.axhspan(expected - half, expected + half, color=RED, alpha=0.12, label="95% (pointwise)")
    ax.set_xlabel("PIT  =  P(Y $\\leq$ y$_{true}$ | x)")
    ax.set_ylabel("count")
    ax.legend(fontsize=7, loc="lower right")

    headline = "\n".join(textwrap.wrap(f"PIT histogram — {_interpretation(summary)}", 52))
    ax.set_title(
        f"{headline}\n"
        f"mean={summary['mean']:.4f} ({UNIFORM_MEAN})  var={summary['var']:.4f} ({UNIFORM_VAR:.4f})",
        fontsize=8,
        loc="left",
    )


def _plot_ecdf_diff(ax, pit: np.ndarray, summary: dict[str, float], n_rows: int) -> None:
    n = len(pit)
    u = np.sort(pit)
    ecdf = np.arange(1, n + 1) / n
    # Simultaneous 95% band from the asymptotic Kolmogorov distribution.
    band = np.sqrt(-0.5 * np.log(0.05 / 2) / n)
    ax.axhline(0, color=RED, lw=1.2)
    ax.fill_between([0, 1], -band, band, color=RED, alpha=0.12, label="95% (simultaneous)")
    ax.step(u, ecdf - u, where="post", color=BLUE, lw=1.4)
    ax.set_xlabel("PIT")
    ax.set_ylabel("ECDF $-$ uniform")
    ax.legend(fontsize=7, loc="lower right")

    ax.set_title(
        f"KS vs uniform (median of {summary['repeats']:.0f}): "
        f"D={summary['ks_stat']:.4f}  p={summary['ks_pval']:.3g}\n"
        f"n={summary['n']:.0f} independent theta of {n_rows} rows; "
        f"rejects at 5% in {summary['reject_frac']:.0%} of subsamples",
        fontsize=8,
        loc="left",
    )


def _plot_dap(ax, dap: np.ndarray, prior: np.ndarray, target_name: str, bins: int, ks) -> None:
    lo = min(dap.min(), prior.min())
    hi = max(dap.max(), prior.max())
    edges = np.linspace(np.log10(lo), np.log10(hi), bins + 1)
    ax.hist(np.log10(prior), bins=edges, color=GREY, alpha=0.65, label="prior (true theta)")
    ax.hist(
        np.log10(dap), bins=edges, histtype="step", color=BLUE, lw=1.6,
        label="data-averaged posterior",
    )
    ax.set_xlabel(f"log10({target_name})")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    ax.set_title(
        f"Data-averaged posterior vs prior\n"
        f"2-sample KS: D={ks.statistic:.4f}  p={ks.pvalue:.3g}",
        fontsize=8,
        loc="left",
    )


def run_sbc(
    artifact: RQSArtifact,
    records: Records,
    split: str,
    target_name: str,
    bins: int,
    rng: np.random.Generator,
    path_png: Path,
    repeats: int = 200,
) -> dict[str, float]:
    """Compute and plot SBC diagnostics for one split. Returns the independent-subset stats."""
    X_raw = np.stack([rec["x_raw"] for rec in records]).astype(np.float32)
    y_true = _column(records, "y_raw")

    # Exact rank statistic: the artifact rescales X and y with its own fitted scalers.
    pit = np.asarray(artifact.cdf(X_raw, y_true.astype(np.float32)), dtype=np.float64)
    all_stats = _uniformity_stats(pit)
    indep_stats, indep = _repeated_uniformity(pit, _sibling_groups(records), rng, repeats)

    # One posterior draw per independent theta; pooled, they must reproduce the prior.
    dap = np.asarray(artifact.sample(X_raw[indep], rng), dtype=np.float64)
    dap_ks = stats.ks_2samp(np.log10(dap), np.log10(y_true[indep]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _plot_pit_hist(axes[0], pit[indep], bins, indep_stats)
    _plot_ecdf_diff(axes[1], pit[indep], indep_stats, len(records))
    _plot_dap(axes[2], dap, y_true[indep], target_name, bins, dap_ks)
    fig.suptitle(f"{split} split — simulation-based calibration of {target_name}")
    fig.tight_layout()
    _save(fig, path_png)

    log.info(
        "SBC %s — independent n=%d (median of %d subsamples, rejects in %.0f%%): "
        "KS D=%.4f p=%.3g, PIT mean=%.4f var=%.4f → %s",
        split,
        indep_stats["n"],
        repeats,
        100 * indep_stats["reject_frac"],
        indep_stats["ks_stat"],
        indep_stats["ks_pval"],
        indep_stats["mean"],
        indep_stats["var"],
        _interpretation(indep_stats),
    )
    log.info(
        "SBC %s — all rows n=%d (correlated siblings): KS D=%.4f p=%.3g",
        split,
        all_stats["n"],
        all_stats["ks_stat"],
        all_stats["ks_pval"],
    )
    log.info(
        "DAP %s — pooled posterior vs prior: KS D=%.4f p=%.3g",
        split,
        dap_ks.statistic,
        dap_ks.pvalue,
    )
    return indep_stats


# ------------------------------------------------------------------------- main


def _save(fig, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote %s", path_png)


def _parse_args():
    parser = argparse.ArgumentParser(description="Validate the posterior against the prior")
    parser.add_argument("--data_file", help="Path to the data file", default="datasets/estimint_simulations_y9.parquet")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--predictor", type=str, help="Predictor name", default="prev_y9")
    parser.add_argument("--target", type=str, help="Target variable name", default="eir")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=["test"],
        help="Splits to evaluate (default: test)",
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--calib_frac", type=float, default=0.06, help="Calibration fraction when creating a split")
    parser.add_argument("--stratify", action="store_true", help="Stratify a newly created split by target magnitude")
    parser.add_argument("--sbc", action="store_true", help="Run simulation-based calibration against a trained model")
    parser.add_argument("--no-prior", dest="prior", action="store_false", help="Skip the prior histograms")
    parser.add_argument("--model", type=str, default="dide-ic/estiMINT", help="Hugging Face repo id or local artifact dir")
    parser.add_argument("--revision", type=str, default=None, help="Model revision (default: latest)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write plots and fitted scalers (default: posterior_outputs/<predictor>-<target>)",
    )
    return parser.parse_args()


def _load_artifact(args) -> RQSArtifact:
    log.info("Loading model %s (%s)", args.model, args.revision or "latest")
    artifact = ConditionalRQS.from_pretrained(
        args.model, predictor=args.predictor, target=args.target, revision=args.revision
    )
    expected = get_features(args.predictor)
    if artifact.feature_names != expected:
        raise ValueError(
            f"Model features {artifact.feature_names} do not match the prepared feature order {expected}."
        )
    return artifact


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    name = f"{args.predictor}-{args.target}"
    output_dir = Path(args.output_dir or f"posterior_outputs/{name}")

    raw_df = duckdb.read_parquet(args.data_file).df()
    cfg = DictConfig(
        {
            "seed": args.seed,
            "data_file": args.data_file,
            "predictor": args.predictor,
            "target": args.target,
            "split_file": f"datasets/split_{name}.csv",
            "use_existing_split": True,
            "stratify": args.stratify,
            "output_dir": str(output_dir),
        }
    )
    prepared_data = prepare_data(raw_df, cfg, calib_frac=args.calib_frac)
    artifact = _load_artifact(args) if args.sbc else None

    for split in args.splits:
        records = _split_records(prepared_data, split)
        if not records:
            log.warning("Split %s is empty; skipping", split)
            continue

        if args.prior:
            targets = _split_targets(records)
            log.info("%s split %s:\n%s", args.target, split, _describe(targets["raw"]))
            plot_prior(
                split, targets, args.target, args.bins, output_dir / f"prior_{args.target}_{split}.png"
            )

        if artifact is not None:
            # Seeded per split so results do not depend on which splits ran before.
            rng = np.random.default_rng([args.seed, SPLIT_NAMES.index(split)])
            run_sbc(
                artifact,
                records,
                split,
                args.target,
                min(args.bins, 20),
                rng,
                output_dir / f"sbc_{args.target}_{split}.png",
            )


if __name__ == "__main__":
    main()
