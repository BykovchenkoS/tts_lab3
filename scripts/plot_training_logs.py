import re
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass


def parse_tensor_value(s):
    s = s.strip()
    m = re.match(r"tensor\(\s*([0-9.eE+\-]+)", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return None


def parse_log(filepath):
    data = defaultdict(list)
    current_global_step = None
    current_step = None
    metrics_this_step = {}
    steps_per_epoch = None

    global_pattern = re.compile(r"GLOBAL_STEP:\s*(\d+)", re.IGNORECASE)
    step_pattern = re.compile(r"STEP:\s*(\d+)/(\d+)", re.IGNORECASE)
    metric_pattern = re.compile(
        r"\|\s*>\s*([\w_]+):\s*(.+?)(?:\s+\((.+?)\))?\s*$"
    )

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            gs = global_pattern.search(line)
            si = step_pattern.search(line)

            if gs:
                new_global_step = int(gs.group(1))

                if si:
                    steps_per_epoch = int(si.group(2))
                    current_step = int(si.group(1))

                if metrics_this_step and current_global_step is not None:
                    for key, val in metrics_this_step.items():
                        data[key].append(val)
                    data["epoch"].append(0)
                    data["global_step"].append(current_global_step)
                    data["step_in_epoch"].append(
                        current_step if current_step is not None else 0
                    )
                    metrics_this_step = {}

                current_global_step = new_global_step

            elif "| >" in line:
                m = metric_pattern.search(line)
                if m:
                    key = m.group(1).strip()
                    raw_val = m.group(2).strip()
                    val = parse_tensor_value(raw_val)
                    if val is not None:
                        if m.group(3):
                            avg_val = parse_tensor_value(m.group(3).strip())
                            if avg_val is not None:
                                val = avg_val
                        metrics_this_step[key] = val

    if metrics_this_step and current_global_step is not None:
        for key, val in metrics_this_step.items():
            data[key].append(val)
        data["epoch"].append(0)
        data["global_step"].append(current_global_step)
        data["step_in_epoch"].append(
            current_step if current_step is not None else 0
        )

    if not data["epoch"]:
        return {}

    min_len = len(data["epoch"])
    for key in list(data.keys()):
        if key in ("epoch", "global_step", "step_in_epoch"):
            continue
        if len(data[key]) < min_len:
            min_len = len(data[key])
    for key in list(data.keys()):
        if len(data[key]) > min_len:
            data[key] = data[key][:min_len]

    data["x"] = data["global_step"]

    if steps_per_epoch:
        print(f"  Steps per epoch: {steps_per_epoch}")
        for i in range(len(data["epoch"])):
            data["epoch"][i] = int(data["x"][i] // steps_per_epoch)

    return dict(data)


def smooth(y, window=20):
    if len(y) < window:
        return y
    out = []
    for i in range(len(y)):
        s = max(0, i - window // 2)
        e = min(len(y), i + window // 2 + 1)
        out.append(sum(y[s:e]) / (e - s))
    return out


def fix_origin_zero(ax):
    xticks = ax.get_xticks()
    if len(xticks) > 0 and xticks[0] == 0:
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        if len(xticklabels) > 0 and xticklabels[0] == "0":
            xticklabels[0] = ""
            ax.set_xticklabels(xticklabels)


def plot_metric(data, keys, filename, title, ylabel, colors=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    if colors is None:
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]
    x = data.get("x", list(range(len(data.get(keys[0], [])))))
    for i, key in enumerate(keys):
        values = data.get(key, [])
        if not values:
            continue
        c = colors[i % len(colors)]
        ax.plot(x, values, color=c, alpha=0.15, linewidth=0.5)
        ax.plot(x, smooth(values), color=c, linewidth=2, label=key)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0, y=0)
    fix_origin_zero(ax)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="best", fontsize=11)
    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(filename)}")


def make_boxplot(epoch_dict, all_epochs, output_dir):
    fig, ax = plt.subplots(figsize=(max(20, len(all_epochs) * 0.18), 7))
    box_data = [epoch_dict[ep] for ep in all_epochs]
    labels = [str(ep) for ep in all_epochs]

    bp = ax.boxplot(box_data, tick_labels=labels,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=1, alpha=0.2),
                    whiskerprops=dict(linewidth=0.8),
                    boxprops=dict(linewidth=0.8),
                    medianprops=dict(color="#e74c3c", linewidth=1.5),
                    capprops=dict(linewidth=0.8))

    for j, patch in enumerate(bp["boxes"]):
        t = j / max(1, len(bp["boxes"]) - 1)
        r = 0.83 * (1 - t) + 0.20 * t
        g = 0.28 * (1 - t) + 0.60 * t
        b = 0.24 * (1 - t) + 0.86 * t
        patch.set_facecolor((r, g, b, 0.75))

    ax.set_title("Loss Distribution by Epoch (Full Overview)", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontsize=14, labelpad=10)
    ax.set_ylabel("Loss", fontsize=14, labelpad=10)

    ticks = list(range(0, len(labels), 10))
    if (len(labels) - 1) not in ticks:
        ticks.append(len(labels) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], fontsize=10)
    fix_origin_zero(ax)

    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(False)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)

    plt.tight_layout()
    out = os.path.join(output_dir, "09_loss_boxplot_FULL.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  09_loss_boxplot_FULL.png ({len(all_epochs)} epochs)")

    mid = (len(all_epochs) + 1) // 2
    halves = [
        (all_epochs[:mid],
         "09a_loss_boxplot_detail_0-{}.png".format(all_epochs[mid - 1]),
         "Loss Distribution by Epoch (Detail: 0 - {})".format(all_epochs[mid - 1])),
        (all_epochs[mid:],
         "09b_loss_boxplot_detail_{}-{}.png".format(all_epochs[mid], all_epochs[-1]),
         "Loss Distribution by Epoch (Detail: {} - {})".format(all_epochs[mid], all_epochs[-1])),
    ]

    for epochs_subset, filename, title in halves:
        if len(epochs_subset) < 2:
            continue

        box_data = [epoch_dict[ep] for ep in epochs_subset]
        labels = [str(ep) for ep in epochs_subset]

        fig, ax = plt.subplots(figsize=(max(14, len(epochs_subset) * 0.35), 7))

        bp = ax.boxplot(box_data, tick_labels=labels,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=2, alpha=0.3),
                        whiskerprops=dict(linewidth=1.2),
                        boxprops=dict(linewidth=1.2),
                        medianprops=dict(color="#e74c3c", linewidth=2),
                        capprops=dict(linewidth=1.2))

        for j, patch in enumerate(bp["boxes"]):
            t = j / max(1, len(bp["boxes"]) - 1)
            r = 0.83 * (1 - t) + 0.20 * t
            g = 0.28 * (1 - t) + 0.60 * t
            b = 0.24 * (1 - t) + 0.86 * t
            patch.set_facecolor((r, g, b, 0.75))

        ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("Epoch", fontsize=14, labelpad=10)
        ax.set_ylabel("Loss", fontsize=14, labelpad=10)

        n_labels = len(labels)
        if n_labels > 40:
            show_every = 5
        elif n_labels > 20:
            show_every = 2
        else:
            show_every = 1

        ticks = list(range(0, n_labels, show_every))
        if (n_labels - 1) not in ticks:
            ticks.append(n_labels - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], fontsize=11)
        fix_origin_zero(ax)

        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {filename} ({len(epochs_subset)} epochs)")


def make_avg_loss_line(epoch_dict, all_epochs, output_dir):
    avg_loss_all = [sum(epoch_dict[ep]) / len(epoch_dict[ep]) for ep in all_epochs]

    fig, ax = plt.subplots(figsize=(max(16, len(all_epochs) * 0.12), 6))
    ax.plot(all_epochs, avg_loss_all, color="#e74c3c", linewidth=2.5,
            marker="o", markersize=2)
    ax.fill_between(all_epochs, avg_loss_all, alpha=0.15, color="#e74c3c")

    ax.set_title("Average Loss per Epoch (Full Overview)", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontsize=14, labelpad=10)
    ax.set_ylabel("Average Loss", fontsize=14, labelpad=10)

    ticks = list(range(0, len(all_epochs), 10))
    if (len(all_epochs) - 1) not in ticks:
        ticks.append(len(all_epochs) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(all_epochs[i]) for i in ticks], fontsize=10)
    fix_origin_zero(ax)

    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    plt.tight_layout()
    out = os.path.join(output_dir, "10_avg_loss_per_epoch_FULL.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  10_avg_loss_per_epoch_FULL.png ({len(all_epochs)} epochs)")

    mid = (len(all_epochs) + 1) // 2
    halves = [
        (all_epochs[:mid],
         "10a_avg_loss_detail_0-{}.png".format(all_epochs[mid - 1]),
         "Average Loss per Epoch (Detail: 0 - {})".format(all_epochs[mid - 1])),
        (all_epochs[mid:],
         "10b_avg_loss_detail_{}-{}.png".format(all_epochs[mid], all_epochs[-1]),
         "Average Loss per Epoch (Detail: {} - {})".format(all_epochs[mid], all_epochs[-1])),
    ]

    for epochs_subset, filename, title in halves:
        if len(epochs_subset) < 2:
            continue

        avg_loss = [sum(epoch_dict[ep]) / len(epoch_dict[ep]) for ep in epochs_subset]

        fig, ax = plt.subplots(figsize=(max(14, len(epochs_subset) * 0.35), 6))
        ax.plot(epochs_subset, avg_loss, color="#e74c3c", linewidth=2.5,
                marker="o", markersize=4, markerfacecolor="#c0392b",
                markeredgecolor="white", markeredgewidth=0.5)
        ax.fill_between(epochs_subset, avg_loss, alpha=0.15, color="#e74c3c")

        ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("Epoch", fontsize=14, labelpad=10)
        ax.set_ylabel("Average Loss", fontsize=14, labelpad=10)

        n_labels = len(epochs_subset)
        if n_labels > 40:
            show_every = 5
        elif n_labels > 20:
            show_every = 2
        else:
            show_every = 1

        ticks = list(range(0, n_labels, show_every))
        if (n_labels - 1) not in ticks:
            ticks.append(n_labels - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(epochs_subset[i]) for i in ticks], fontsize=11)
        fix_origin_zero(ax)

        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {filename} ({len(epochs_subset)} epochs)")


def generate_plots(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n = len(data["epoch"])
    epochs = max(data["epoch"]) + 1
    print(f"Parsed: {n} points, {epochs} epochs")

    skip = {"epoch", "global_step", "step_in_epoch", "x"}
    metrics = sorted(k for k in data.keys() if k not in skip)
    print(f"Metrics: {metrics}")

    if "loss" in data:
        plot_metric(data, ["loss"],
            os.path.join(output_dir, "01_total_loss.png"),
            "Total Training Loss", "Loss", ["#e74c3c"])

    comp = [k for k in ["decoder_loss", "postnet_loss", "stopnet_loss"] if k in data]
    if comp:
        plot_metric(data, comp,
            os.path.join(output_dir, "02_component_losses.png"),
            "Component Losses (Decoder / Postnet / Stopnet)", "Loss",
            ["#3498db", "#2ecc71", "#e67e22"])

    if "align_error" in data:
        plot_metric(data, ["align_error"],
            os.path.join(output_dir, "03_alignment_error.png"),
            "Attention Alignment Error", "Alignment Error", ["#9b59b6"])

    lr = [k for k in ["current_lr", "lr"] if k in data]
    if lr:
        plot_metric(data, lr,
            os.path.join(output_dir, "04_learning_rate.png"),
            "Learning Rate Schedule", "Learning Rate", ["#1abc9c"])

    if "grad_norm" in data:
        plot_metric(data, ["grad_norm"],
            os.path.join(output_dir, "05_gradient_norm.png"),
            "Gradient Norm", "Gradient Norm", ["#e67e22"])

    all_l = [k for k in ["loss", "decoder_loss", "postnet_loss"] if k in data]
    if len(all_l) >= 2:
        plot_metric(data, all_l,
            os.path.join(output_dir, "06_all_losses_combined.png"),
            "All Losses Combined", "Loss",
            ["#e74c3c", "#3498db", "#2ecc71", "#e67e22"])

    ssim = [k for k in ["decoder_ssim_loss", "postnet_ssim_loss"] if k in data]
    if ssim:
        plot_metric(data, ssim,
            os.path.join(output_dir, "07_ssim_loss.png"),
            "SSIM Loss (Decoder / Postnet)", "SSIM Loss",
            ["#8e44ad", "#2980b9"])

    ds = [k for k in ["decoder_diff_spec_loss", "postnet_diff_spec_loss"] if k in data]
    if ds:
        plot_metric(data, ds,
            os.path.join(output_dir, "08_diff_spec_loss.png"),
            "Spectral Difference Loss (Decoder / Postnet)", "Diff Spec Loss",
            ["#c0392b", "#27ae60"])

    if "loss" in data and len(set(data["epoch"])) > 3:
        epoch_dict = defaultdict(list)
        for i, ep in enumerate(data["epoch"]):
            epoch_dict[ep].append(data["loss"][i])
        all_epochs = sorted(epoch_dict.keys())
        make_boxplot(epoch_dict, all_epochs, output_dir)
        make_avg_loss_line(epoch_dict, all_epochs, output_dir)

    generate_summary(data, output_dir)


def generate_summary(data, output_dir):
    path = os.path.join(output_dir, "training_summary.txt")
    lines = ["=" * 60, "TRAINING SUMMARY", "=" * 60, ""]
    epochs = max(data["epoch"]) + 1
    lines.append(f"Total data points: {len(data['epoch'])}")
    lines.append(f"Total epochs:      {epochs}")

    info = [
        ("loss", "Total Loss"), ("decoder_loss", "Decoder Loss"),
        ("postnet_loss", "Postnet Loss"), ("stopnet_loss", "Stopnet Loss"),
        ("align_error", "Alignment Error"), ("grad_norm", "Gradient Norm"),
        ("current_lr", "Learning Rate"), ("lr", "Learning Rate"),
    ]

    lines += ["", "-" * 60, "METRIC STATISTICS", "-" * 60]
    for key, name in info:
        vals = data.get(key, [])
        if not vals:
            continue
        n20 = max(1, len(vals) // 20)
        sa = sum(vals[:n20]) / n20
        ea = sum(vals[-n20:]) / n20
        imp = ((sa - ea) / sa * 100) if sa > 0 else 0
        bi = vals.index(min(vals))
        be = data["epoch"][bi] if bi < len(data["epoch"]) else "?"
        bs = data["x"][bi] if bi < len(data["x"]) else "?"
        lines += ["", f"  {name} ({key}):",
                  f"    Min:          {min(vals):.6f}",
                  f"    Max:          {max(vals):.6f}",
                  f"    Final:        {vals[-1]:.6f}",
                  f"    Start avg:    {sa:.6f}",
                  f"    End avg:      {ea:.6f}",
                  f"    Improvement:  {imp:+.1f}%",
                  f"    Best:         epoch {be}, step {bs} = {min(vals):.6f}"]

    lines += ["", "-" * 60, "OVERALL ASSESSMENT", "-" * 60]
    lv = data.get("loss", [])
    if lv:
        n20 = max(1, len(lv) // 20)
        s, e = sum(lv[:n20]) / n20, sum(lv[-n20:]) / n20
        if e < s * 0.5:
            lines.append("  [GOOD] Loss decreased by >50%")
        elif e < s * 0.75:
            lines.append("  [OK] Loss decreased by 25-50%")
        elif e < s:
            lines.append("  [WEAK] Loss decreased but <25%")
        else:
            lines.append("  [BAD] Loss NOT decreasing!")
    sn = data.get("stopnet_loss", [])
    if sn:
        a = sum(sn[-10:]) / min(10, len(sn))
        if a < 0.01:
            lines.append(f"  [GOOD] Stopnet loss very low: {a:.4f}")
        elif a < 0.05:
            lines.append(f"  [OK] Stopnet loss moderate: {a:.4f}")
        else:
            lines.append(f"  [BAD] Stopnet loss too high: {a:.4f}")
    ae = data.get("align_error", [])
    if ae:
        n20 = max(1, len(ae) // 20)
        s, e = sum(ae[:n20]) / n20, sum(ae[-n20:]) / n20
        if e < s * 0.7:
            lines.append("  [GOOD] Alignment error decreasing")
        elif e < s:
            lines.append("  [OK] Alignment error slowly decreasing")
        else:
            lines.append("  [BAD] Alignment error NOT decreasing!")

    lines += ["", "=" * 60]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  training_summary.txt")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:")
        print(f"  python {sys.argv[0]} <trainer_0_log.txt> [--output_dir <dir>]")
        sys.exit(1)
    log_path = sys.argv[1]
    if not os.path.isfile(log_path):
        print(f"ERROR: File not found: {log_path}")
        sys.exit(1)
    output_dir = "graphs"
    if "--output_dir" in sys.argv:
        idx = sys.argv.index("--output_dir")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    print(f"Log:     {log_path}")
    print(f"Output:  {output_dir}\n")
    data = parse_log(log_path)
    if not data:
        print("ERROR: No data parsed!")
        sys.exit(1)
    generate_plots(data, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
