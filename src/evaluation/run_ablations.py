"""Run ablation experiments to measure the impact of each training component.
Compares: no augmentation, no warmup, no hard mining, no cosine annealing, and full pipeline.
Generates comparison plots at the end.

Usage: PYTHONPATH="." python src/evaluation/run_ablations.py
       PYTHONPATH="." python src/evaluation/run_ablations.py --quick  (3 epochs per variant)
"""
import os, sys, subprocess, json, glob, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
OUT = "reports/ablations"
os.makedirs(OUT, exist_ok=True)

BASE_ARGS = [
    sys.executable, "src/training/train_face_cnn.py",
    "--data", "data/face/widerface",
    "--batch-size", "128",
    "--lr", "0.001",
    "--epochs", "3",  # override via --quick or --epochs
]

VARIANTS = {
    "full": {
        "label": "Full pipeline",
        "args": [],
        "color": "#1f77b4",
    },
    "no_aug": {
        "label": "No augmentation",
        "desc": "Removes flip, rotation, brightness, contrast augmentation",
        "args": ["--no-augment"],
        "color": "#ff7f0e",
    },
    "no_warmup": {
        "label": "No LR warmup",
        "desc": "Starts at full learning rate from epoch 1",
        "args": ["--warmup", "0"],
        "color": "#2ca02c",
    },
    "no_cosine": {
        "label": "Step decay (no cosine)",
        "desc": "ReduceLROnPlateau instead of CosineAnnealingLR",
        "args": ["--no-cosine"],
        "color": "#d62728",
    },
    "no_hardmine": {
        "label": "No hard-negative mining",
        "desc": "Disables every-3rd-epoch hard negative mining",
        "args": ["--no-hardmine"],
        "color": "#9467bd",
    },
    "lr_5x": {
        "label": "LR=0.005 (5×)",
        "desc": "Higher learning rate",
        "args": ["--lr", "0.005"],
        "color": "#8c564b",
    },
    "batch_64": {
        "label": "Batch size 64",
        "desc": "Half batch size, double gradients per epoch",
        "args": ["--batch-size", "64"],
        "color": "#e377c2",
    },
}

RESULTS_FILE = os.path.join(OUT, "results.json")


def train_variant(name, cfg, epochs):
    args = BASE_ARGS + cfg["args"] + ["--epochs", str(epochs)]
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "face_cnn.pth")
    csv_path = os.path.join(out_dir, "metrics.csv")
    args += ["--output", out_path, "--log-csv", csv_path]

    print(f"\n{'='*60}")
    print(f"Running: {cfg['label']}")
    print(f"  {cfg.get('desc', '')}")
    print(f"  {' '.join(args)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(args, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        print(result.stderr[-500:])
        return None

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_val = df["val_obj_loss"].min() if "val_obj_loss" in df.columns else None
        best_f1 = df["val_f1"].max() if "val_f1" in df.columns else None
    else:
        best_val = None
        best_f1 = None

    return {
        "variant": name,
        "label": cfg["label"],
        "epochs": epochs,
        "time_s": round(elapsed, 1),
        "final_val_loss": float(df["val_obj_loss"].iloc[-1]) if best_val is not None else None,
        "best_val_loss": float(best_val) if best_val is not None else None,
        "best_f1": float(best_f1) if best_f1 is not None else None,
        "metrics_csv": csv_path,
        "returncode": result.returncode,
    }


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [r["label"] for r in results if r is not None]
    losses = [r["best_val_loss"] for r in results if r is not None]
    f1s = [r["best_f1"] for r in results if r is not None]
    times = [r["time_s"] for r in results if r is not None]
    colors = [VARIANTS[r["variant"]]["color"] for r in results if r is not None]

    axes[0].barh(names, losses, color=colors)
    axes[0].set_xlabel("Best Validation Loss")
    axes[0].set_title("Detection Quality")

    axes[1].barh(names, f1s, color=colors)
    axes[1].set_xlabel("Best Validation F1")
    axes[1].set_title("F1 Score")

    axes[2].barh(names, times, color=colors)
    axes[2].set_xlabel("Training Time (s)")
    axes[2].set_title("Computational Cost")

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Training Ablation Study", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {os.path.join(OUT, 'ablation_comparison.pdf')}")


def plot_loss_curves(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        if r is None or not os.path.exists(r["metrics_csv"]):
            continue
        df = pd.read_csv(r["metrics_csv"])
        if "val_obj_loss" in df.columns:
            ax.plot(df["epoch"], df["val_obj_loss"],
                    label=r["label"], color=VARIANTS[r["variant"]]["color"],
                    marker=".", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Curves by Variant")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_loss_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved to {os.path.join(OUT, 'ablation_loss_curves.pdf')}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs per variant (default: 20, use --quick for 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 3 epochs per variant for quick smoke test")
    parser.add_argument("--variants", nargs="+",
                        default=list(VARIANTS.keys()),
                        help=f"Variants to run: {list(VARIANTS.keys())}")
    args = parser.parse_args()

    epochs = 3 if args.quick else args.epochs
    print(f"Ablation study: {len(args.variants)} variants × {epochs} epochs each")

    results = []
    for name in args.variants:
        if name not in VARIANTS:
            print(f"Unknown variant: {name}, skipping")
            continue
        r = train_variant(name, VARIANTS[name], epochs)
        results.append(r)
        # Save intermediate results
        with open(RESULTS_FILE, "w") as f:
            json.dump([r for r in results if r], f, indent=2)

    results = [r for r in results if r]
    if results:
        plot_results(results)
        plot_loss_curves(results)
        print(f"\nAll results saved to {OUT}/")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
