"""Generate all training analysis plots from metrics CSV + analysis JSONs.
Usage: PYTHONPATH="." python src/evaluation/plot_training.py
       PYTHONPATH="." python src/evaluation/plot_training.py --metrics models/training_metrics.csv
"""

import os, sys, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "lines.linewidth": 1.8,
})

OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_metrics(csv_path):
    df = pd.read_csv(csv_path)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_analysis_jsons(analysis_dir):
    """Load all epoch_XX_analysis.json files into a dict indexed by epoch."""
    data = {}
    paths = sorted(glob.glob(os.path.join(analysis_dir, "epoch_*_analysis.json")))
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        data[d["epoch"]] = d
    return data


# ── Plotting Functions ──────────────────────────────────────────────


def plot_loss_curves(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train Total", color="#1f77b4")
    ax.plot(df["epoch"], df["train_obj_loss"], label="Train Obj", color="#ff7f0e", alpha=0.7)
    ax.plot(df["epoch"], df["train_bbox_loss"], label="Train BBox (×5)", color="#2ca02c", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend(); ax.set_yscale("log" if df["train_loss"].max() > 10 * df["train_loss"].min() else "linear")

    ax = axes[1]
    ax.plot(df["epoch"], df["val_obj_loss"], label="Val Objectness", color="#d62728", marker="o", ms=3)
    ax.plot(df["epoch"], df["val_bbox_loss"], label="Val BBox", color="#9467bd", marker="s", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Validation Loss Curves")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_loss_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  01_loss_curves.pdf")


def plot_lr_schedule(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["lr"], color="#e377c2", linewidth=2.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_lr_schedule.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  02_lr_schedule.pdf")


def plot_prf1_curves(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["val_precision"], label="Precision", color="#2ca02c")
    ax.plot(df["epoch"], df["val_recall"], label="Recall", color="#1f77b4")
    ax.plot(df["epoch"], df["val_f1"], label="F1 Score", color="#d62728", linewidth=2.5)
    ax.plot(df["epoch"], df["val_specificity"], label="Specificity", color="#9467bd", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation Precision / Recall / F1")
    ax.legend(); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_prf1_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  03_prf1_curves.pdf")


def plot_iou_curves(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["epoch"], df["val_mean_iou"], label="Mean IoU", color="#1f77b4", marker="o", ms=3)
    ax.plot(df["epoch"], df["val_iou_at_05"], label="IoU > 0.5", color="#ff7f0e", marker="s", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
    ax.set_title("Bounding Box IoU Metrics")
    ax.legend(); ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.plot(df["epoch"], df["mean_dx_err"], label="Center X Error", color="#d62728")
    ax.plot(df["epoch"], df["mean_dy_err"], label="Center Y Error", color="#2ca02c")
    ax.plot(df["epoch"], df["mean_ls_err"], label="Log-Size Error", color="#9467bd")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Absolute Error")
    ax.set_title("Bounding Box Regression Errors")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_iou_bbox_errors.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  04_iou_bbox_errors.pdf")


def plot_calibration(df, analysis_data, metrics_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["epoch"], df["val_ece"], color="#e377c2", linewidth=2.5, marker="o", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("ECE")
    ax.set_title("Expected Calibration Error over Training")
    ax.set_ylim(0, min(1, df["val_ece"].max() * 1.5 + 0.02))

    ax = axes[1]
    # Reliability diagram from last available confidence JSON
    conf_paths = sorted(glob.glob(os.path.join(
        os.path.dirname(metrics_path),
        "face_cnn_analysis/epoch_*_confidence.json"
    )))
    if conf_paths:
        with open(conf_paths[-1]) as f:
            conf = json.load(f)
        bins = conf.get("bin_centers", [])
        counts = conf.get("bin_counts", [])
        avg_confs = conf.get("bin_avg_confidences", [])
        if bins and counts:
            width = 1.0 / len(bins)
            ax.bar(bins, counts, width=width, alpha=0.6, color="#1f77b4", label="% of predictions")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            if avg_confs:
                ax.scatter(bins, avg_confs, color="#d62728", s=15, zorder=5, label="Avg confidence")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Frequency / Accuracy")
    ax.set_title("Reliability Diagram (Last Epoch)")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_calibration.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  05_calibration.pdf")


def plot_training_stability(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["epoch"], df["mean_weight_cosine_sim"], color="#2ca02c", linewidth=2, marker="o", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cosine Similarity")
    ax.set_title("Weight Stability (Epoch-over-Epoch Cosine Sim)")
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    if "mean_l1_ratio" in df.columns:
        ax.plot(df["epoch"], df["mean_l1_ratio"], color="#d62728", linewidth=2, marker="s", ms=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("L1 Ratio")
        ax.set_title("Weight Change Magnitude (Epoch-over-Epoch L1)")
        ax.set_yscale("log" if df["mean_l1_ratio"].max() > 10 * df["mean_l1_ratio"].min() else "linear")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_training_stability.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  06_training_stability.pdf")


def plot_gradient_norm(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["epoch"], df["grad_norm"], color="#1f77b4", linewidth=2, marker="o", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Total Gradient Norm")
    ax.set_yscale("log")

    ax = axes[1]
    if "max_update_ratio" in df.columns:
        ax.plot(df["epoch"], df["max_update_ratio"], color="#ff7f0e", linewidth=2, marker="s", ms=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Update Ratio")
        ax.set_title("Max Gradient-to-Weight Ratio (Signal-to-Noise)")
        ax.set_yscale("log" if df["max_update_ratio"].max() > 10 * df["max_update_ratio"].min() else "linear")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_gradient_norms.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  07_gradient_norms.pdf")


def plot_data_efficiency(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["epoch"], df["epoch_time_s"], label="Epoch Time", color="#1f77b4", marker="o", ms=4)
    ax.plot(df["epoch"], df["gpu_mem_mb"], label="GPU Memory", color="#d62728", marker="s", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s) / Memory (MB)")
    ax.set_title("Training Speed & Memory")
    ax.legend()

    ax = axes[1]
    if "data_load_pct" in df.columns:
        labels = ["Data Loading", "Compute"]
        mid_epoch = len(df) // 2
        sizes = [df["data_load_pct"].iloc[mid_epoch], df["compute_pct"].iloc[mid_epoch]]
        colors = ["#ff7f0e", "#1f77b4"]
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors, startangle=90)
        ax.set_title(f"Epoch Time Breakdown (Epoch {mid_epoch+1})")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "08_data_efficiency.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  08_data_efficiency.pdf")


def plot_flops_breakdown():
    """Per-layer FLOPs pie chart from theoretical computation."""
    sizes = [39.3, 37.7, 37.7, 37.7, 0.26]
    labels = ["Block1 Conv\n(3→16, 5×5)", "Block2 Conv\n(16→32, 3×3)",
              "Block3 Conv\n(32→64, 3×3)", "Block4 Conv\n(64→128, 3×3)",
              "Head Conv\n(128→4, 1×1)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.78,
        textprops={"fontsize": 9})
    for t in autotexts: t.set_fontweight("bold")
    ax.set_title("Theoretical FLOPs Distribution by Layer\n(Forward, 128×128 input)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "09_flops_breakdown.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  09_flops_breakdown.pdf")


def plot_inference_flops():
    scales = [1.0, 0.87, 0.76, 0.66, 0.57]
    gflops = [2.86, 2.16, 1.64, 1.24, 0.93]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(scales)), gflops, color="#1f77b4", width=0.6)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"{s:.2f}×" for s in scales])
    ax.set_xlabel("Scale Factor"); ax.set_ylabel("GFLOPs")
    ax.set_title("Inference FLOPs per Scale (640×480 frame)")
    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(0.5, 0.95, f"Total: {sum(gflops):.2f} GFLOPs",
            transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "10_inference_flops.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  10_inference_flops.pdf")


def plot_dead_neurons(df):
    if "dead_neuron_pct" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["dead_neuron_pct"], color="#d62728", linewidth=2, marker="o", ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weights Near Zero (%)")
    ax.set_title("% of Conv Weights with |w| < 0.01 (Near-Dead Neurons)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "11_dead_neurons.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  11_dead_neurons.pdf")


def plot_effective_lr(analysis_data):
    if not analysis_data:
        return
    last_epoch = max(analysis_data.keys())
    eff_lr = analysis_data[last_epoch].get("effective_lr", {})
    if not eff_lr:
        return
    names = []
    values = []
    for k, v in eff_lr.items():
        names.append(k)
        values.append(v.get("effective_lr", 0))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(range(len(names)), values, color="#2ca02c", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Effective Step Size")
    ax.set_title(f"AdamW Effective Learning Rate per Layer (Epoch {last_epoch})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "12_effective_lr.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  12_effective_lr.pdf")


def plot_spectral_analysis(analysis_data):
    if not analysis_data:
        return
    epochs = sorted(analysis_data.keys())
    layers = set()
    for e in epochs:
        ws = analysis_data[e].get("weight_stats", {})
        for k in ws:
            if k != "_total" and "spectral_norm" in ws[k]:
                layers.add(k)
    if not layers:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for layer in sorted(layers):
        sn = [analysis_data[e]["weight_stats"][layer]["spectral_norm"] for e in epochs if layer in analysis_data[e].get("weight_stats", {})]
        er = [analysis_data[e]["weight_stats"][layer]["effective_rank"] for e in epochs if layer in analysis_data[e].get("weight_stats", {})]
        axes[0].plot(epochs[:len(sn)], sn, label=layer, marker=".", ms=3)
        axes[1].plot(epochs[:len(er)], er, label=layer, marker=".", ms=3)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Spectral Norm")
    axes[0].set_title("Weight Matrix Spectral Norm"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Effective Rank")
    axes[1].set_title("Weight Matrix Effective Rank"); axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "13_spectral_analysis.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  13_spectral_analysis.pdf")


def plot_summary_dashboard(df):
    """Single comprehensive dashboard figure."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Loss
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(df["epoch"], df["train_loss"], label="Train Total", color="#1f77b4")
    ax.plot(df["epoch"], df["val_obj_loss"], label="Val Objectness", color="#d62728")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves"); ax.legend()

    # 2. P/R/F1
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(df["epoch"], df["val_precision"], label="P", color="#2ca02c")
    ax.plot(df["epoch"], df["val_recall"], label="R", color="#1f77b4")
    ax.plot(df["epoch"], df["val_f1"], label="F1", color="#d62728", linewidth=2.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("P / R / F1"); ax.legend(); ax.set_ylim(0, 1)

    # 3. IoU
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(df["epoch"], df["val_mean_iou"], label="Mean IoU", color="#1f77b4")
    ax.plot(df["epoch"], df["val_iou_at_05"], label="IoU>0.5", color="#ff7f0e")
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
    ax.set_title("BBox IoU"); ax.legend(); ax.set_ylim(0, 1)

    # 4. Time & Memory
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(df["epoch"], df["epoch_time_s"], label="Time", color="#1f77b4")
    ax_twin = ax.twinx()
    ax_twin.plot(df["epoch"], df["gpu_mem_mb"], label="Memory", color="#d62728")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Time (s)", color="#1f77b4")
    ax_twin.set_ylabel("Mem (MB)", color="#d62728")
    ax.set_title("Time & GPU Memory")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    # 5. Learning Rate
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df["epoch"], df["lr"], color="#e377c2", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule")

    # 6. Gradient Norm
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(df["epoch"], df["grad_norm"], color="#1f77b4")
    ax.set_xlabel("Epoch"); ax.set_ylabel("||∇||")
    ax.set_title("Gradient Norm")
    ax.set_yscale("log")

    # 7. Weight Stability
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(df["epoch"], df["mean_weight_cosine_sim"], color="#2ca02c")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cos Sim")
    ax.set_title("Weight Stability"); ax.set_ylim(0, 1.05)

    # 8. Calibration
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(df["epoch"], df["val_ece"], color="#9467bd", marker="o", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("ECE")
    ax.set_title("Expected Calibration Error")

    # 9. Dead Neurons
    if "dead_neuron_pct" in df.columns:
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(df["epoch"], df["dead_neuron_pct"], color="#d62728")
        ax.set_xlabel("Epoch"); ax.set_ylabel("% Near Zero")
        ax.set_title("Near-Dead Weights")

    # 10. ECE + time
    if "data_load_pct" in df.columns:
        ax = fig.add_subplot(gs[2, 2])
        mid = len(df) // 2
        sizes = [df["data_load_pct"].iloc[mid], df["compute_pct"].iloc[mid]]
        ax.pie(sizes, labels=["Load", "Compute"], autopct="%1.0f%%",
               colors=["#ff7f0e", "#1f77b4"], startangle=90)
        ax.set_title("I/O vs Compute Split")

    # 11. Key metrics table
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    last = df.iloc[-1]
    first = df.iloc[0]
    metrics_text = (
        f"Training Summary\n"
        f"{'─'*25}\n"
        f"Epochs:         {int(last['epoch'])}\n"
        f"Total time:     {df['epoch_time_s'].sum()/60:.1f}m\n"
        f"Best val F1:    {df['val_f1'].max():.3f}\n"
        f"Best val IoU:   {df['val_mean_iou'].max():.3f}\n"
        f"Final val loss: {last['val_obj_loss']:.4f}\n"
        f"Best ECE:       {df['val_ece'].min():.4f}\n"
        f"Total FLOPs:    {df['cumulative_tflops'].iloc[-1]:.1f}T\n"
        f"Start loss:     {first['train_loss']:.2f} → {last['train_loss']:.2f}\n"
    )
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="center", fontfamily="monospace")
    ax.set_title("Key Metrics")

    fig.suptitle("FaceCNN Training — Full Dashboard", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUT_DIR, "00_dashboard.pdf"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  00_dashboard.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="models/training_metrics.csv")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        print(f"Metrics file not found: {args.metrics}")
        print("Run training first: python src/training/train_face_cnn.py")
        sys.exit(1)

    print(f"Loading metrics from {args.metrics}")
    df = load_metrics(args.metrics)
    print(f"  {len(df)} epochs loaded")

    if args.analysis_dir is None:
        args.analysis_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.metrics)),
            "face_cnn_analysis"
        )
    analysis_data = load_analysis_jsons(args.analysis_dir) if os.path.isdir(args.analysis_dir) else {}
    print(f"  {len(analysis_data)} analysis epochs loaded")

    print("\nGenerating plots...")
    plot_loss_curves(df)
    plot_lr_schedule(df)
    plot_prf1_curves(df)
    plot_iou_curves(df)
    plot_calibration(df, analysis_data, args.metrics)
    plot_training_stability(df)
    plot_gradient_norm(df)
    plot_data_efficiency(df)
    plot_flops_breakdown()
    plot_inference_flops()
    plot_dead_neurons(df)
    plot_effective_lr(analysis_data)
    plot_spectral_analysis(analysis_data)
    plot_summary_dashboard(df)

    print(f"\nAll plots saved to {OUT_DIR}/")
    print(f"Files: {sorted(os.listdir(OUT_DIR))}")


if __name__ == "__main__":
    main()
