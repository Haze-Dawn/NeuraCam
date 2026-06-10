"""
v7 Benchmark Plot Generator
============================
Generates publication-quality plots from v7 benchmark JSON results.

Reads the benchmark JSON output and produces:
  1. mAP comparison bar chart (overall + per-category + per-size)
  2. Latency/FPS comparison
  3. PR curve
  4. Threshold sweep
  5. NMS sweep
  6. Resolution sensitivity
  7. Confusion matrix heatmap
  8. Deployment speed comparison

Usage:
  python src/evaluation/generate_v7_benchmark_plots.py benchmarks/v7_comprehensive/benchmark_*.json
  python src/evaluation/generate_v7_benchmark_plots.py benchmarks/v7_comprehensive/benchmark_*.json --output plots/
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")


# Color palette (colorblind-friendly)
COLORS = {
    "v7":      "#2196F3",
    "v7_p4only": "#FF9800",
    "easy":    "#4CAF50",
    "medium":  "#FF9800",
    "hard":    "#F44336",
    "small":   "#9C27B0",
    "medium_face": "#2196F3",
    "large":   "#4CAF50",
    "onnx_fp32": "#2196F3",
    "onnx_int8": "#FF5722",
    "openvino":  "#4CAF50",
    "tensorrt":  "#9C27B0",
}


def set_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_map_comparison(results, output_dir):
    """Bar chart: overall mAP@0.5, mAP(COCO), per-category, per-size."""
    models = list(results.keys())
    names = [results[m]["model_name"].split("(")[0].strip() for m in models]
    x = np.arange(len(models))
    width = 0.25
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Overall mAP
    ax = axes[0]
    map50 = [results[m]["mAP@0.5"] for m in models]
    mapcoco = [results[m]["mAP_coco"] for m in models]
    bars1 = ax.bar(x - width/2, map50, width, label="mAP@0.5", color=COLORS["v7"])
    bars2 = ax.bar(x + width/2, mapcoco, width, label="mAP(COCO)", color=COLORS["v7_p4only"])
    ax.set_ylabel("mAP")
    ax.set_title("Overall mAP")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    for bar, val in zip(bars1, map50):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, mapcoco):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    
    # Per-category
    ax = axes[1]
    categories = ["easy", "medium", "hard"]
    cat_width = 0.25
    for i, cat in enumerate(categories):
        vals = [results[m]["per_category"][cat]["mAP@0.5"] for m in models]
        ax.bar(x + (i - 1) * cat_width, vals, cat_width,
               label=cat.capitalize(), color=COLORS[cat])
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Per-Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    
    # Per-size
    ax = axes[2]
    sizes = ["small", "medium", "large"]
    size_labels = ["Small\n(<32px)", "Medium\n(32-96px)", "Large\n(>96px)"]
    for i, sz in enumerate(sizes):
        vals = [results[m]["per_size"][sz]["mAP@0.5"] for m in models]
        ax.bar(x + (i - 1) * cat_width, vals, cat_width,
               label=size_labels[i], color=COLORS[sz])
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Per-Size")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    
    plt.suptitle("FaceFCN v7 — mAP Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "map_comparison.png"))
    plt.close()


def plot_latency(results, output_dir):
    """Bar chart: latency + FPS."""
    models = list(results.keys())
    names = [results[m]["model_name"].split("(")[0].strip() for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    mean_ms = [results[m]["speed"]["mean_ms"] for m in models]
    p95_ms = [results[m]["speed"]["p95_ms"] for m in models]
    fps = [results[m]["speed"]["fps"] for m in models]
    x = np.arange(len(models))
    
    bars1 = ax1.bar(x - 0.15, mean_ms, 0.3, label="Mean", color=COLORS["v7"])
    bars2 = ax1.bar(x + 0.15, p95_ms, 0.3, label="P95", color=COLORS["v7_p4only"])
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Inference Latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.legend()
    for bar, val in zip(bars1, mean_ms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    
    bars = ax2.bar(x, fps, 0.5, color=[COLORS["v7"], COLORS["v7_p4only"]][:len(models)])
    ax2.set_ylabel("FPS")
    ax2.set_title("Throughput")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    for bar, val in zip(bars, fps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    
    plt.suptitle("FaceFCN v7 — Latency & Throughput", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_fps.png"))
    plt.close()


def plot_pr_curves(results, output_dir):
    """PR curves from saved PR data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for key, r in results.items():
        pr = r.get("pr_curve", {})
        if not pr:
            continue
        recalls = np.array(pr["recalls"])
        precisions = np.array(pr["precisions"])
        color = COLORS.get(key, "#666666")
        ax.plot(recalls, precisions, color=color, lw=2,
                label=f'{r["model_name"].split("(")[0].strip()} '
                      f'(mAP={r["mAP@0.5"]:.3f})')
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()


def plot_threshold_sweep(results, output_dir):
    """Threshold sweep: mAP vs confidence threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for key, r in results.items():
        sweep = r.get("threshold_sweep", {})
        if not sweep:
            continue
        thresholds = sorted(sweep.keys())
        maps = [sweep[t]["mAP@0.5"] for t in thresholds]
        color = COLORS.get(key, "#666666")
        ax.plot(thresholds, maps, "o-", color=color, lw=2, ms=4,
                label=r["model_name"].split("(")[0].strip())
    
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Confidence Threshold Sensitivity")
    ax.legend()
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_sweep.png"))
    plt.close()


def plot_nms_sweep(results, output_dir):
    """NMS sweep: mAP vs NMS IoU threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for key, r in results.items():
        sweep = r.get("nms_sweep", {})
        if not sweep:
            continue
        thresholds = sorted(sweep.keys())
        maps = [sweep[t]["mAP@0.5"] for t in thresholds]
        color = COLORS.get(key, "#666666")
        ax.plot(thresholds, maps, "s-", color=color, lw=2, ms=4,
                label=r["model_name"].split("(")[0].strip())
    
    ax.set_xlabel("NMS IoU Threshold")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("NMS IoU Threshold Sensitivity")
    ax.legend()
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nms_sweep.png"))
    plt.close()


def plot_resolution_sensitivity(results, output_dir):
    """Resolution sensitivity: mAP and FPS vs input resolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for key, r in results.items():
        res = r.get("resolution_sensitivity", {})
        if not res:
            continue
        resolutions = sorted(res.keys())
        maps = [res[rez]["mAP@0.5"] for rez in resolutions]
        fps_vals = [res[rez]["fps"] for rez in resolutions]
        labels = resolutions
        color = COLORS.get(key, "#666666")
        ax1.plot(range(len(labels)), maps, "o-", color=color, lw=2,
                 label=r["model_name"].split("(")[0].strip())
        ax2.plot(range(len(labels)), fps_vals, "s-", color=color, lw=2,
                 label=r["model_name"].split("(")[0].strip())
    
    for ax, ylabel in [(ax1, "mAP@0.5"), (ax2, "FPS")]:
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        # Resolution labels would need data from the results
        ax.set_xlabel("Input Resolution")
    
    plt.suptitle("Resolution Sensitivity", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resolution_sensitivity.png"))
    plt.close()


def plot_confusion_matrix(results, output_dir):
    """Confusion matrix heatmap."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for ax, (key, r) in zip(axes, results.items()):
        c = r.get("confusion", {})
        tp, fp, fn = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0)
        matrix = np.array([[tp, fp], [fn, 0]])
        
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["TP", "FP"])
        ax.set_yticklabels(["Detected", "Missed"])
        ax.set_title(r["model_name"].split("(")[0].strip())
        
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:,}", ha="center", va="center",
                            color="white" if val > matrix.max() / 2 else "black",
                            fontsize=12, fontweight="bold")
    
    plt.suptitle("Confusion Matrix", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def plot_deployment(results, output_dir):
    """Deployment speed comparison across backends."""
    has_deploy = any("deployment" in r for r in results.values())
    if not has_deploy:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = list(results.keys())
    backends = set()
    for r in results.values():
        backends.update(r.get("deployment", {}).keys())
    backends = sorted(backends)
    
    x = np.arange(len(models))
    width = 0.8 / max(len(backends), 1)
    
    for i, backend in enumerate(backends):
        fps_vals = []
        for m in models:
            dep = results[m].get("deployment", {}).get(backend)
            fps_vals.append(dep.get("fps", 0) if dep else 0)
        color = COLORS.get(backend, f"C{i}")
        bars = ax.bar(x + (i - len(backends)/2 + 0.5) * width, fps_vals, width,
                       label=backend, color=color)
        for bar, val in zip(bars, fps_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_ylabel("FPS")
    ax.set_title("Deployment Backend Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([results[m]["model_name"].split("(")[0].strip() for m in models],
                        rotation=15, ha="right")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "deployment_comparison.png"))
    plt.close()


def plot_event_categories(results, output_dir):
    """Top event categories per model."""
    for key, r in results.items():
        pe = r.get("per_event", {})
        if not pe:
            continue
        
        sorted_ev = sorted(pe.items(), key=lambda x: x[1]["mAP@0.5"], reverse=True)[:15]
        cats = [e[0].split("--", 1)[-1].replace("_", " ") for e in sorted_ev]
        maps = [e[1]["mAP@0.5"] for e in sorted_ev]
        gts = [e[1]["n_gt"] for e in sorted_ev]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        y = np.arange(len(cats))
        color = COLORS.get(key, "#2196F3")
        ax1.barh(y, maps, color=color, alpha=0.8)
        ax1.set_yticks(y)
        ax1.set_yticklabels(cats, fontsize=9)
        ax1.set_xlabel("mAP@0.5")
        ax1.set_title(f'{r["model_name"].split("(")[0].strip()} — Top Categories')
        ax1.invert_yaxis()
        
        ax2.barh(y, gts, color=color, alpha=0.6)
        ax2.set_yticks(y)
        ax2.set_yticklabels(cats, fontsize=9)
        ax2.set_xlabel("GT Faces")
        ax2.set_title("Ground Truth Count")
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"event_categories_{key}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate v7 benchmark plots")
    parser.add_argument("json_path", help="Benchmark JSON results file")
    parser.add_argument("--output", default=None, help="Output directory for plots")
    args = parser.parse_args()
    
    if not HAS_MPL:
        print("Cannot generate plots without matplotlib.")
        sys.exit(1)
    
    set_style()
    
    with open(args.json_path) as f:
        results = json.load(f)
    
    output_dir = args.output or os.path.join(os.path.dirname(args.json_path), "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating plots from: {args.json_path}")
    print(f"Output directory: {output_dir}")
    
    plot_map_comparison(results, output_dir)
    print(f"  [OK] mAP comparison")
    
    plot_latency(results, output_dir)
    print(f"  [OK] Latency/FPS")
    
    plot_pr_curves(results, output_dir)
    print(f"  [OK] PR curves")
    
    plot_confusion_matrix(results, output_dir)
    print(f"  [OK] Confusion matrix")
    
    if any("threshold_sweep" in r for r in results.values()):
        plot_threshold_sweep(results, output_dir)
        print(f"  [OK] Threshold sweep")
    
    if any("nms_sweep" in r for r in results.values()):
        plot_nms_sweep(results, output_dir)
        print(f"  [OK] NMS sweep")
    
    if any("resolution_sensitivity" in r for r in results.values()):
        plot_resolution_sensitivity(results, output_dir)
        print(f"  [OK] Resolution sensitivity")
    
    plot_deployment(results, output_dir)
    print(f"  [OK] Deployment comparison")
    
    plot_event_categories(results, output_dir)
    print(f"  [OK] Event categories")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
