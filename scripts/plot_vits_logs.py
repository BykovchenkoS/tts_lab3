"""
Plot VITS training logs (from trainer_0_log.txt).
Based on plot_training_logs.py, adapted for VITS-specific metrics.
"""
import os
import sys
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass

plt.rcParams["axes.unicode_minus"] = False


# в”Ђв”Ђв”Ђ Fonts в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf")
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
except Exception:
    pass


# в”Ђв”Ђв”Ђ Tensor value parser в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def parse_tensor_value(text):
    """Extract float from 'tensor(X, device='cuda:0')' or plain float."""
    text = text.strip()
    m = re.match(r"tensor\(\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", text)
    if m:
        return float(m.group(1))
    try:
        return float(text)
    except ValueError:
        return None


# в”Ђв”Ђв”Ђ Log parser в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def parse_log(filepath):
    """
    Parse VITS trainer_0_log.txt.
    Returns dict: metric_name -> list of values (at each logged GLOBAL_STEP).
    """
    data = {}
    current_step = None
    current_metrics = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Match GLOBAL_STEP line
            gs_match = re.search(r"GLOBAL_STEP:\s*(\d+)", line)
            if gs_match:
                # Flush previous step
                if current_step is not None and current_metrics:
                    for key, val in current_metrics.items():
                        if key not in data:
                            data[key] = []
                        data[key].append(val)

                current_step = int(gs_match.group(1))
                current_metrics = {}
                continue

            # Match metric line:  | > metric_name: value  (avg_value)
            if current_step is not None and "| > " in line:
                m = re.match(r"\s*\|\s*>\s*(\S+):\s*(.*?)\s*\(", line.strip())
                if not m:
                    # Try without parentheses (e.g. current_lr)
                    m2 = re.match(r"\s*\|\s*>\s*(\S+):\s*(.*?)\s*$", line.strip())
                    if m2:
                        key = m2.group(1)
                        val = parse_tensor_value(m2.group(2))
                        if val is not None:
                            current_metrics[key] = val
                    continue

                key = m.group(1)
                raw_val = m.group(2).strip()
                val = parse_tensor_value(raw_val)
                if val is not None:
                    current_metrics[key] = val

    # Flush last step
    if current_step is not None and current_metrics:
        for key, val in current_metrics.items():
            if key not in data:
                data[key] = []
            data[key].append(val)

    return data


# в”Ђв”Ђв”Ђ Helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def fix_origin_zero(ax):
    """Hide duplicate '0' tick label on X axis (origin already on Y)."""
    xticks = ax.get_xticks()
    labels = [str(int(l)) if l == int(l) else f"{l:.1g}" for l in xticks]
    if len(labels) > 1 and labels[0] == "0":
        labels[0] = ""
    ax.set_xticklabels(labels)


def get_steps_per_epoch(data, step_col="current_lr_0"):
    """Estimate steps per epoch from LR drops (StepLR halves LR)."""
    lrs = data.get(step_col, [])
    if not lrs:
        return 205  # default for LJSpeech

    # Find when LR drops (StepLR gamma=0.5)
    initial_lr = lrs[0]
    for i, lr in enumerate(lrs):
        if lr < initial_lr * 0.9:  # drop detected
            return i + 1
    return 205


def uniformize_series(data, keys):
    """Trim all series to the same length."""
    lengths = [len(data.get(k, [])) for k in keys if k in data]
    if not lengths:
        return {}
    min_len = min(lengths)
    result = {}
    for k in keys:
        if k in data:
            result[k] = data[k][:min_len]
    return result


# в”Ђв”Ђв”Ђ Plot functions в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
COLORS = {
    "disc": "#e74c3c",
    "gen": "#3498db",
    "mel": "#2ecc71",
    "kl": "#9b59b6",
    "feat": "#f39c12",
    "dur": "#1abc9c",
    "lr": "#e67e22",
    "gn": "#34495e",
}


def make_line(ax, x, y, color, label, linewidth=1.5, alpha=0.9):
    """Plot a line with proper styling."""
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label)


def plot_losses(data, output_dir, steps_per_epoch):
    """Plot generator and discriminator losses."""
    metrics = uniformize_series(data, ["loss_1", "loss_disc", "loss_gen",
                                        "loss_mel", "loss_kl", "loss_feat",
                                        "loss_duration"])
    if not metrics:
        return

    x = np.arange(len(metrics.get("loss_1", []))) / steps_per_epoch

    # в”Ђв”Ђ 1. Total losses в”Ђв”Ђ
    fig, ax = plt.subplots(figsize=(14, 6))
    if "loss_1" in metrics:
        make_line(ax, x, metrics["loss_1"], COLORS["gen"], "Generator total (loss_1)")
    if "loss_disc" in metrics:
        make_line(ax, x, metrics["loss_disc"], COLORS["disc"], "Discriminator (loss_disc)")
    ax.set_title("VITS: Generator vs Discriminator Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", fontsize=10)
    ax.margins(x=0, y=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fix_origin_zero(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vits_01_total_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # в”Ђв”Ђ 2. Generator sub-losses в”Ђв”Ђ
    fig, ax = plt.subplots(figsize=(14, 6))
    if "loss_gen" in metrics:
        make_line(ax, x, metrics["loss_gen"], COLORS["gen"], "Adversarial (loss_gen)")
    if "loss_mel" in metrics:
        make_line(ax, x, metrics["loss_mel"], COLORS["mel"], "Mel-spectrogram (loss_mel)")
    if "loss_kl" in metrics:
        make_line(ax, x, metrics["loss_kl"], COLORS["kl"], "KL divergence (loss_kl)")
    if "loss_feat" in metrics:
        make_line(ax, x, metrics["loss_feat"], COLORS["feat"], "Feature matching (loss_feat)")
    if "loss_duration" in metrics:
        make_line(ax, x, metrics["loss_duration"], COLORS["dur"], "Duration (loss_duration)")
    ax.set_title("VITS: Generator Sub-Losses", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", fontsize=10)
    ax.margins(x=0, y=0)
    ax.set_xlim(left=0)
    fix_origin_zero(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vits_02_gen_losses.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # в”Ђв”Ђ 3. Mel loss (detail) в”Ђв”Ђ
    fig, ax = plt.subplots(figsize=(14, 6))
    if "loss_mel" in metrics:
        make_line(ax, x, metrics["loss_mel"], COLORS["mel"], "Mel loss")
    ax.set_title("VITS: Mel-Spectrogram Loss (Detail)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", fontsize=10)
    ax.margins(x=0, y=0)
    ax.set_xlim(left=0)
    fix_origin_zero(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vits_03_mel_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  vits_01_total_loss.png")
    print("  vits_02_gen_losses.png")
    print("  vits_03_mel_loss.png")


def plot_grad_norms(data, output_dir, steps_per_epoch):
    """Plot discriminator and generator gradient norms."""
    metrics = uniformize_series(data, ["grad_norm_0", "grad_norm_1"])
    if not metrics:
        return

    x = np.arange(len(metrics.get("grad_norm_0", []))) / steps_per_epoch

    fig, ax = plt.subplots(figsize=(14, 6))
    if "grad_norm_0" in metrics:
        make_line(ax, x, metrics["grad_norm_0"], COLORS["disc"], "Discriminator (grad_norm_0)")
    if "grad_norm_1" in metrics:
        make_line(ax, x, metrics["grad_norm_1"], COLORS["gen"], "Generator (grad_norm_1)")
    ax.set_title("VITS: Gradient Norms", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Grad Norm")
    ax.legend(loc="best", fontsize=10)
    ax.margins(x=0, y=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fix_origin_zero(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vits_04_grad_norm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  vits_04_grad_norm.png")


def plot_lr(data, output_dir, steps_per_epoch):
    """Plot learning rates (discriminator and generator)."""
    metrics = uniformize_series(data, ["current_lr_0", "current_lr_1"])
    if not metrics:
        return

    x = np.arange(len(metrics.get("current_lr_0", []))) / steps_per_epoch

    fig, ax = plt.subplots(figsize=(14, 6))
    if "current_lr_0" in metrics:
        make_line(ax, x, metrics["current_lr_0"], COLORS["disc"], "Discriminator LR (lr_0)")
    if "current_lr_1" in metrics:
        make_line(ax, x, metrics["current_lr_1"], COLORS["gen"], "Generator LR (lr_1)")
    ax.set_title("VITS: Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend(loc="best", fontsize=10)
    ax.margins(x=0, y=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fix_origin_zero(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vits_05_lr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  vits_05_lr.png")


def plot_timing(data, output_dir, steps_per_epoch):
    """Plot step time and loader time."""
    metrics = uniformize_series(data, ["step_time", "loader_time"])
    if not metrics:
        return

    x = np.arange(len(metrics.get("step_time", []))) / steps_per_epoch

    fig, ax = plt.subplots(figsize=(14, 6))
    if "step_time" in metrics:
        make_line(ax, x, metrics["step_time"], "#8e44ad", "Step time (s)")
    if "loader_time" in metrics:
        make_line(ax, x, metrics["loader_time"], "#16a085", "Loader time (s)")
    ax.set_title("VITS: Step / Loader Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.legend(loc="best", fontsize=10)
    ax.margins(x=0, y=0)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fix_origin_zero(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vits_06_timing.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  vits_06_timing.png")


def plot_boxplots(data, output_dir):
    """Boxplots for all numeric metrics."""
    skip_keys = {"step_time", "loader_time", "current_lr_0", "current_lr_1",
                 "loss_disc_real_0", "loss_disc_real_1", "loss_disc_real_2",
                 "loss_disc_real_3", "loss_disc_real_4", "loss_disc_real_5"}
    plot_keys = [k for k in data if k not in skip_keys and len(data[k]) > 1]

    if not plot_keys:
        return

    # Select key metrics for the boxplot
    key_metrics = ["loss_1", "loss_disc", "loss_gen", "loss_mel", "loss_kl",
                   "loss_feat", "loss_duration", "grad_norm_0", "grad_norm_1"]
    plot_keys = [k for k in key_metrics if k in data]
    if not plot_keys:
        plot_keys = [k for k in data if k not in skip_keys and len(data[k]) > 1]

    n = len(plot_keys)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = [COLORS.get("gen", "#3498db"), COLORS.get("disc", "#e74c3c"),
              COLORS.get("mel", "#2ecc71"), COLORS.get("kl", "#9b59b6"),
              COLORS.get("feat", "#f39c12"), COLORS.get("dur", "#1abc9c"),
              COLORS.get("gn", "#34495e"), "#e67e22", "#1abc9c"]

    for i, key in enumerate(plot_keys):
        vals = data[key]
        bp = axes[i].boxplot(vals, patch_artist=True, showfliers=True)
        bp["boxes"][0].set_facecolor(colors[i % len(colors)])
        bp["boxes"][0].set_alpha(0.7)
        axes[i].set_title(key.replace("_", " ").title(), fontsize=9)
        axes[i].set_ylabel("Value", fontsize=8)

    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "vits_07_boxplots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  vits_07_boxplots.png")


def generate_summary(data, steps_per_epoch, output_path):
    """Generate text summary of VITS training."""
    lines = ["=" * 60, "VITS TRAINING SUMMARY", "=" * 60, ""]

    total_points = max(len(v) for v in data.values()) if data else 0
    total_epochs = total_points / steps_per_epoch
    lines.append(f"Total logged points: {total_points}")
    lines.append(f"Estimated epochs:    {total_epochs:.1f}")
    lines.append(f"Steps per epoch:     {steps_per_epoch}")
    lines.append("")

    # Key metrics summary
    key_metrics = {
        "loss_1": "Generator Total Loss",
        "loss_disc": "Discriminator Loss",
        "loss_gen": "Generator Adversarial Loss",
        "loss_mel": "Mel-Spectrogram Loss",
        "loss_kl": "KL Divergence Loss",
        "loss_feat": "Feature Matching Loss",
        "loss_duration": "Duration Loss",
        "grad_norm_0": "Discriminator Grad Norm",
        "grad_norm_1": "Generator Grad Norm",
    }

    lines.append("--- Key Metrics ---")
    for key, name in key_metrics.items():
        if key in data and len(data[key]) > 0:
            vals = data[key]
            first = vals[0]
            last = vals[-1]
            best = min(vals) if "loss" in key.lower() else max(vals)
            improvement = (first - last) / first * 100 if first != 0 else 0
            lines.append(f"  {name}:")
            lines.append(f"    Start: {first:.4f}  ->  End: {last:.4f}  (best: {best:.4f})")
            lines.append(f"    Improvement: {improvement:+.1f}%")

    lines.append("")
    lines.append("=" * 60)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  vits_training_summary.txt")


# в”Ђв”Ђв”Ђ Main в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def main():
    parser = argparse.ArgumentParser(description="Plot VITS training logs")
    parser.add_argument("--log_dir", default="run/vits",
                        help="Path to VITS run directory")
    parser.add_argument("--output_dir", default="graphs",
                        help="Output directory for plots")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="Override steps per epoch (auto-detected if not set)")
    args = parser.parse_args()

    # Find log file
    import glob as globmod
    log_files = globmod.glob(os.path.join(args.log_dir, "*", "trainer_0_log.txt"))
    if not log_files:
        # Try direct path
        if os.path.isfile(os.path.join(args.log_dir, "trainer_0_log.txt")):
            log_files = [os.path.join(args.log_dir, "trainer_0_log.txt")]

    if not log_files:
        print(f"[ERROR] No trainer_0_log.txt found in {args.log_dir}")
        sys.exit(1)

    log_file = log_files[0]
    print(f"Log file: {log_file}")

    # Parse
    print("Parsing log...")
    data = parse_log(log_file)

    print(f"Found {len(data)} metrics:")
    for key in sorted(data.keys()):
        print(f"  {key}: {len(data[key])} points")

    # Steps per epoch
    if args.steps_per_epoch:
        spe = args.steps_per_epoch
    else:
        spe = get_steps_per_epoch(data)
    print(f"Steps per epoch: {spe}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot
    print("\nGenerating plots...")
    plot_losses(data, args.output_dir, spe)
    plot_grad_norms(data, args.output_dir, spe)
    plot_lr(data, args.output_dir, spe)
    plot_timing(data, args.output_dir, spe)
    plot_boxplots(data, args.output_dir)
    generate_summary(data, spe, os.path.join(args.output_dir, "vits_training_summary.txt"))

    print("\nDone!")


if __name__ == "__main__":
    main()

