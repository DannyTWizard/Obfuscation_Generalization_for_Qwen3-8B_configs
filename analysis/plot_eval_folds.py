import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_file(csv_path: Path, output_dir: Path | None = None) -> None:
    """
    For a given eval CSV, plot reward_hacking_rate and penalisation_rate
    as a function of artifact_step for each (data, eval_fold) pair.
    One figure is produced per (file, data, eval_fold).
    """
    if output_dir is None:
        output_dir = csv_path.parent

    df = pd.read_csv(csv_path)

    required_cols = {
        "artifact_step",
        "data",
        "eval_fold",
        "reward_hacking_rate",
        "penalisation_rate",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    # Ensure sorted by step within each (data, fold) for nice lines
    df = df.sort_values(["data", "eval_fold", "artifact_step"])

    for (data_name, fold), fold_df in df.groupby(["data", "eval_fold"]):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            fold_df["artifact_step"],
            fold_df["reward_hacking_rate"],
            marker="o",
            label="reward_hacking_rate",
        )
        ax.plot(
            fold_df["artifact_step"],
            fold_df["penalisation_rate"],
            marker="s",
            label="penalisation_rate",
        )

        ax.set_xlabel("artifact_step")
        ax.set_ylabel("rate")
        ax.set_title(f"{csv_path.name} — {data_name} — {fold}")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_fold = str(fold).replace("/", "_")
        safe_data = str(data_name).replace("/", "_")
        out_name = f"{csv_path.stem}_data-{safe_data}_fold-{safe_fold}.png"
        out_path = output_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot reward_hacking_rate and penalisation_rate vs artifact_step "
            "for each eval_fold in one or more eval_results CSVs."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help=(
            "Paths to eval CSV files. If omitted, defaults to the three "
            "standard files in metrics/eval/: eval_results_24/42/50.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write PNGs into. "
            "Defaults to the directory of each CSV."
        ),
    )

    args = parser.parse_args()

    if args.csv_files:
        csv_paths = args.csv_files
    else:
        # Default to the three known files the user mentioned
        repo_root = Path(__file__).resolve().parents[1]
        default_dir = repo_root / "metrics" / "eval"
        csv_paths = [
            default_dir / "eval_results_24.csv",
            default_dir / "eval_results_42.csv",
            default_dir / "eval_results_50.csv",
        ]

    for csv_path in csv_paths:
        if not csv_path.is_file():
            print(f"Skipping {csv_path} (not found)")
            continue
        print(f"Processing {csv_path}...")
        if args.output_dir is None:
            plot_file(csv_path)
        else:
            plot_file(csv_path, args.output_dir)


if __name__ == "__main__":
    main()
