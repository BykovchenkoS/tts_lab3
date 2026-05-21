import os
import sys
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display


plt.rcParams["axes.unicode_minus"] = False
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass


def load_audio(filepath, sr=22050):
    return librosa.load(filepath, sr=sr)[0]


def extract_mfcc(wav, sr=22050, n_mfcc=13):
    return librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)


def extract_f0(wav, sr=22050, fmin=80, fmax=400):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        wav, fmin=fmin, fmax=fmax, sr=sr
    )
    return f0  # np.array with NaN for unvoiced


def extract_spectral_centroid(wav, sr=22050):
    return librosa.feature.spectral_centroid(y=wav, sr=sr)[0]


def extract_spectral_bandwidth(wav, sr=22050):
    return librosa.feature.spectral_bandwidth(y=wav, sr=sr)[0]


def extract_spectral_rolloff(wav, sr=22050):
    return librosa.feature.spectral_rolloff(y=wav, sr=sr)[0]


def extract_spectral_flatness(wav, sr=22050):
    return librosa.feature.spectral_flatness(y=wav)[0]


def extract_zero_crossing_rate(wav):
    return librosa.feature.zero_crossing_rate(wav)[0]


def find_pairs(eval_dir, ljspeech_dir=None):
    """Find synth_*.wav and match with originals."""
    synth_files = sorted(glob.glob(os.path.join(eval_dir, "synth_*.wav")))
    pairs = []

    for synth_path in synth_files:
        basename = os.path.basename(synth_path)  # synth_LJ042-0149.wav
        # Extract LJ id
        lj_id = basename.replace("synth_", "").replace(".wav", "")  # LJ042-0149

        if ljspeech_dir:
            ref_path = os.path.join(ljspeech_dir, "wavs", lj_id + ".wav")
        else:
            # Try common locations
            for d in ["data/raw/LJSpeech-1.1", "data/LJSpeech-1.1"]:
                ref_path = os.path.join(d, "wavs", lj_id + ".wav")
                if os.path.isfile(ref_path):
                    break

        if os.path.isfile(ref_path):
            pairs.append({
                "id": lj_id,
                "ref": ref_path,
                "synth": synth_path,
            })
        else:
            print(f"  [WARN] Reference not found for {lj_id}")

    return pairs


def plot_mfcc_distribution(ref_mfccs, synth_mfccs, output_path):
    """Plot MFCC coefficient distributions (mean across time for each file)."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("MFCC Coefficient Distributions (Original vs Synthesized)",
                 fontsize=16, fontweight="bold", y=1.02)

    ref_means = np.array([mfcc.mean(axis=1) for mfcc in ref_mfccs])  # (n_files, 13)
    synth_means = np.array([mfcc.mean(axis=1) for mfcc in synth_mfccs])

    for i in range(min(13, ref_means.shape[1])):
        ax = axes[i // 5, i % 5]
        ax.hist(ref_means[:, i], bins=15, alpha=0.6, color="#3498db", label="Original", density=True)
        ax.hist(synth_means[:, i], bins=15, alpha=0.6, color="#e74c3c", label="Synthesized", density=True)
        ax.set_title(f"MFCC-{i}", fontsize=11)
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel("Density", fontsize=10)
        if i >= 10:
            ax.set_xlabel("Value", fontsize=10)

    # Скрыть пустые подграфики (ячейки 13 и 14)
    for i in range(13, 15):
        axes[i // 5, i % 5].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(output_path)}")

def plot_mfcc_heatmaps(ref_mfccs, synth_mfccs, pair_ids, output_path):
    """Plot MFCC heatmaps for first 3 pairs side by side."""
    n_show = min(3, len(ref_mfccs))
    fig, axes = plt.subplots(n_show, 2, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, 2)

    for i in range(n_show):
        librosa.display.specshow(ref_mfccs[i], x_axis="time", ax=axes[i, 0], sr=22050)
        axes[i, 0].set_title(f"Original — {pair_ids[i]}", fontsize=11)
        axes[i, 0].set_ylabel("MFCC Coeff")

        librosa.display.specshow(synth_mfccs[i], x_axis="time", ax=axes[i, 1], sr=22050)
        axes[i, 1].set_title(f"Synthesized — {pair_ids[i]}", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(output_path)}")


def plot_f0_contours(ref_f0s, synth_f0s, pair_ids, output_path):
    """Plot F0 contours for first 5 pairs."""
    n_show = min(5, len(ref_f0s))
    fig, axes = plt.subplots(n_show, 1, figsize=(16, 3 * n_show))
    if n_show == 1:
        axes = np.array([axes])

    for i in range(n_show):
        ref_f0 = ref_f0s[i]
        synth_f0 = synth_f0s[i]

        t_ref = np.arange(len(ref_f0)) / 22050
        t_synth = np.arange(len(synth_f0)) / 22050

        # Mask NaN for plotting
        ref_masked = np.where(np.isnan(ref_f0), None, ref_f0)
        synth_masked = np.where(np.isnan(synth_f0), None, synth_f0)

        axes[i].plot(t_ref, ref_masked, color="#3498db", linewidth=1.5, alpha=0.8, label="Original")
        axes[i].plot(t_synth, synth_masked, color="#e74c3c", linewidth=1.5, alpha=0.8, label="Synthesized")
        axes[i].set_title(f"F0 Contour — {pair_ids[i]}", fontsize=11)
        axes[i].set_ylabel("F0 (Hz)")
        axes[i].legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(output_path)}")


def plot_f0_distribution(ref_f0s, synth_f0s, output_path):
    """Plot F0 distribution (voiced only)."""
    all_ref = []
    all_synth = []
    for f0 in ref_f0s:
        valid = f0[~np.isnan(f0)]
        if len(valid) > 0:
            all_ref.extend(valid.tolist())
    for f0 in synth_f0s:
        valid = f0[~np.isnan(f0)]
        if len(valid) > 0:
            all_synth.extend(valid.tolist())

    if not all_ref or not all_synth:
        print("  [WARN] No valid F0 values")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(all_ref, bins=40, alpha=0.6, color="#3498db", label="Original", density=True)
    ax.hist(all_synth, bins=40, alpha=0.6, color="#e74c3c", label="Synthesized", density=True)
    ax.set_title("F0 Distribution (Original vs Synthesized)", fontsize=16, fontweight="bold")
    ax.set_xlabel("F0 (Hz)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)

    # Stats
    ref_arr = np.array(all_ref)
    synth_arr = np.array(all_synth)
    stats_text = (
        f"Original:    mean={ref_arr.mean():.1f} Hz, std={ref_arr.std():.1f} Hz, median={np.median(ref_arr):.1f} Hz\n"
        f"Synthesized: mean={synth_arr.mean():.1f} Hz, std={synth_arr.std():.1f} Hz, median={np.median(synth_arr):.1f} Hz"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(output_path)}")


def plot_spectral_features(ref_feats, synth_feats, pair_ids, output_path):
    """Plot spectral centroid, bandwidth, rolloff, flatness, ZCR."""
    feature_names = ["spectral_centroid", "spectral_bandwidth",
                     "spectral_rolloff", "spectral_flatness", "zero_crossing_rate"]
    titles = ["Spectral Centroid", "Spectral Bandwidth",
              "Spectral Rolloff", "Spectral Flatness", "Zero Crossing Rate"]
    ylabels = ["Hz", "Hz", "Hz", "", "Rate"]

    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle("Spectral Features (Original vs Synthesized)",
                 fontsize=16, fontweight="bold", y=1.01)

    for fi, fname in enumerate(feature_names):
        ref_vals = [ref_feats[j][fname] for j in range(len(ref_feats))]
        synth_vals = [synth_feats[j][fname] for j in range(len(synth_feats))]

        # Distribution
        ref_flat = np.concatenate([v for v in ref_vals if len(v) > 0])
        synth_flat = np.concatenate([v for v in synth_vals if len(v) > 0])

        if len(ref_flat) > 0 and len(synth_flat) > 0:
            axes[fi, 0].hist(ref_flat, bins=40, alpha=0.6, color="#3498db",
                             label="Original", density=True)
            axes[fi, 0].hist(synth_flat, bins=40, alpha=0.6, color="#e74c3c",
                             label="Synthesized", density=True)
            axes[fi, 0].set_title(f"{titles[fi]} Distribution", fontsize=11)
            axes[fi, 0].legend(fontsize=8)

            # Box plot per file
            ref_per_file = [v.mean() for v in ref_vals if len(v) > 0]
            synth_per_file = [v.mean() for v in synth_vals if len(v) > 0]
            n = min(len(ref_per_file), len(synth_per_file))

            bp = axes[fi, 1].boxplot(
                [ref_per_file[:n], synth_per_file[:n]],
                tick_labels=["Original", "Synthesized"],
                patch_artist=True, showfliers=True
            )
            bp["boxes"][0].set_facecolor("#3498db")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#e74c3c")
            bp["boxes"][1].set_alpha(0.7)
            axes[fi, 1].set_title(f"{titles[fi]} Mean per File", fontsize=11)

        axes[fi, 0].set_ylabel(ylabels[fi], fontsize=10)
        axes[fi, 1].set_ylabel(ylabels[fi], fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(output_path)}")


def generate_stats(ref_mfccs, synth_mfccs, ref_f0s, synth_f0s, ref_feats, synth_feats, output_path):
    """Generate text summary of acoustic features."""
    lines = ["=" * 60, "ACOUSTIC ANALYSIS SUMMARY", "=" * 60, ""]

    # MFCC stats
    lines += ["--- MFCC Coefficients ---"]
    ref_all = np.concatenate([mfcc.mean(axis=1) for mfcc in ref_mfccs])
    synth_all = np.concatenate([mfcc.mean(axis=1) for mfcc in synth_mfccs])
    for i in range(min(13, ref_mfccs[0].shape[0])):
        r_vals = [mfcc[i].mean() for mfcc in ref_mfccs]
        s_vals = [mfcc[i].mean() for mfcc in synth_mfccs]
        lines.append(f"  MFCC-{i:>2}:  Original mean={np.mean(r_vals):>8.4f}  "
                     f"Synth mean={np.mean(s_vals):>8.4f}  "
                     f"Diff={np.mean(s_vals) - np.mean(r_vals):>+8.4f}")

    # F0 stats
    lines += ["", "--- F0 (Pitch) ---"]
    ref_f0_vals = []
    synth_f0_vals = []
    for f0 in ref_f0s:
        valid = f0[~np.isnan(f0)]
        if len(valid) > 0:
            ref_f0_vals.extend(valid.tolist())
    for f0 in synth_f0s:
        valid = f0[~np.isnan(f0)]
        if len(valid) > 0:
            synth_f0_vals.extend(valid.tolist())

    if ref_f0_vals and synth_f0_vals:
        r = np.array(ref_f0_vals)
        s = np.array(synth_f0_vals)
        lines.append(f"  Original:    mean={r.mean():.1f} Hz, std={r.std():.1f}, "
                     f"min={r.min():.1f}, max={r.max():.1f}")
        lines.append(f"  Synthesized: mean={s.mean():.1f} Hz, std={s.std():.1f}, "
                     f"min={s.min():.1f}, max={s.max():.1f}")
        lines.append(f"  F0 difference: {abs(s.mean() - r.mean()):.1f} Hz")

    # Spectral features
    lines += ["", "--- Spectral Features (mean per file) ---"]
    feature_names = ["spectral_centroid", "spectral_bandwidth",
                     "spectral_rolloff", "spectral_flatness", "zero_crossing_rate"]
    for fname in feature_names:
        r_means = [ref_feats[j][fname].mean() for j in range(len(ref_feats))]
        s_means = [synth_feats[j][fname].mean() for j in range(len(synth_feats))]
        lines.append(f"  {fname:<25}:  Orig={np.mean(r_means):>8.2f}  "
                     f"Synth={np.mean(s_means):>8.2f}  "
                     f"Diff={np.mean(s_means) - np.mean(r_means):>+8.2f}")

    lines += ["", "=" * 60]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="output/eval_pairs",
                        help="Directory with synth_*.wav files from evaluate_with_reference.py")
    parser.add_argument("--ljspeech_dir", default=None,
                        help="Path to LJSpeech-1.1 (if not auto-detected)")
    parser.add_argument("--output_dir", default="graphs/acoustic",
                        help="Output directory for graphs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect LJSpeech
    if not args.ljspeech_dir:
        for d in ["data/raw/LJSpeech-1.1", "data/LJSpeech-1.1"]:
            if os.path.isdir(d):
                args.ljspeech_dir = d
                break

    if not args.ljspeech_dir:
        print("[ERROR] LJSpeech directory not found. Use --ljspeech_dir")
        sys.exit(1)

    print("=" * 50)
    print("ACOUSTIC ANALYSIS")
    print("=" * 50)
    print(f"Eval dir:    {args.eval_dir}")
    print(f"LJSpeech:    {args.ljspeech_dir}")
    print(f"Output:      {args.output_dir}\n")

    # Find pairs
    pairs = find_pairs(args.eval_dir, args.ljspeech_dir)
    if not pairs:
        print("[ERROR] No pairs found. Run evaluate_with_reference.py first.")
        sys.exit(1)
    print(f"Found {len(pairs)} pairs\n")

    # Extract features
    ref_mfccs, synth_mfccs = [], []
    ref_f0s, synth_f0s = [], []
    ref_feats, synth_feats = [], []
    pair_ids = []

    for i, pair in enumerate(pairs):
        pair_ids.append(pair["id"])
        print(f"  [{i+1}/{len(pairs)}] {pair['id']}")

        ref_wav = load_audio(pair["ref"])
        synth_wav = load_audio(pair["synth"])

        ref_mfccs.append(extract_mfcc(ref_wav))
        synth_mfccs.append(extract_mfcc(synth_wav))

        ref_f0s.append(extract_f0(ref_wav))
        synth_f0s.append(extract_f0(synth_wav))

        ref_feats.append({
            "spectral_centroid": extract_spectral_centroid(ref_wav),
            "spectral_bandwidth": extract_spectral_bandwidth(ref_wav),
            "spectral_rolloff": extract_spectral_rolloff(ref_wav),
            "spectral_flatness": extract_spectral_flatness(ref_wav),
            "zero_crossing_rate": extract_zero_crossing_rate(ref_wav),
        })
        synth_feats.append({
            "spectral_centroid": extract_spectral_centroid(synth_wav),
            "spectral_bandwidth": extract_spectral_bandwidth(synth_wav),
            "spectral_rolloff": extract_spectral_rolloff(synth_wav),
            "spectral_flatness": extract_spectral_flatness(synth_wav),
            "zero_crossing_rate": extract_zero_crossing_rate(synth_wav),
        })

    print(f"\nGenerating plots...")

    # Generate all plots
    plot_mfcc_distribution(ref_mfccs, synth_mfccs,
                           os.path.join(args.output_dir, "01_mfcc_distribution.png"))
    plot_mfcc_heatmaps(ref_mfccs, synth_mfccs, pair_ids,
                       os.path.join(args.output_dir, "02_mfcc_heatmaps.png"))
    plot_f0_contours(ref_f0s, synth_f0s, pair_ids,
                     os.path.join(args.output_dir, "03_f0_contours.png"))
    plot_f0_distribution(ref_f0s, synth_f0s,
                         os.path.join(args.output_dir, "04_f0_distribution.png"))
    plot_spectral_features(ref_feats, synth_feats, pair_ids,
                           os.path.join(args.output_dir, "05_spectral_features.png"))
    generate_stats(ref_mfccs, synth_mfccs, ref_f0s, synth_f0s, ref_feats, synth_feats,
                   os.path.join(args.output_dir, "acoustic_summary.txt"))

    print("\nDone!")


if __name__ == "__main__":
    main()