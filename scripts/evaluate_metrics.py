import os
import sys
import glob
import argparse
import json
import numpy as np

import librosa
import soundfile as sf


def load_audio(filepath, sr=22050):
    """Load and normalize audio."""
    y, orig_sr = librosa.load(filepath, sr=sr)
    return y


def resample_wav(wav, orig_sr, target_sr):
    """Resample wav to target_sr."""
    return librosa.resample(wav, orig_sr=orig_sr, target_rate=target_sr)


def mel_spectrogram(wav, sr=22050, n_mels=80, n_fft=1024, hop_length=256):
    """Compute mel spectrogram."""
    S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels,
                                        n_fft=n_fft, hop_length=hop_length)
    return S


def align_length(ref, deg):
    """Trim to same length."""
    min_len = min(len(ref), len(deg))
    return ref[:min_len], deg[:min_len]


def compute_pesq(ref, deg, sr=22050):
    """PESQ: -0.5 to 4.5, higher is better."""
    try:
        from pesq import pesq as pesq_fn

        # PESQ requires 16000 Hz
        if sr != 16000:
            ref_16k = resample_wav(ref, sr, 16000)
            deg_16k = resample_wav(deg, sr, 16000)
        else:
            ref_16k, deg_16k = ref, deg

        ref_16k, deg_16k = align_length(ref_16k, deg_16k)

        score = pesq_fn(16000, ref_16k, deg_16k, "wb")
        return float(score)
    except Exception as e:
        print(f"    [WARN] PESQ error: {e}")
        return None


def compute_stoi(ref, deg, sr=22050):
    """STOI: 0.0 to 1.0, higher is better."""
    try:
        from pystoi.stoi import stoi as stoi_fn

        if sr != 16000:
            ref_16k = resample_wav(ref, sr, 16000)
            deg_16k = resample_wav(deg, sr, 16000)
        else:
            ref_16k, deg_16k = ref, deg

        ref_16k, deg_16k = align_length(ref_16k, deg_16k)

        score = stoi_fn(ref_16k, deg_16k, 16000, extended=False)
        return float(score)
    except Exception as e:
        print(f"    [WARN] STOI error: {e}")
        return None


def compute_ssim_mel(ref, deg, sr=22050):
    """Spectral SSIM between mel spectrograms."""
    try:
        from skimage.metrics import structural_similarity as ssim

        mel_ref = mel_spectrogram(ref, sr=sr)
        mel_deg = mel_spectrogram(deg, sr=sr)

        # Align time frames
        min_t = min(mel_ref.shape[1], mel_deg.shape[1])
        mel_ref = mel_ref[:, :min_t]
        mel_deg = mel_deg[:, :min_t]

        score = ssim(mel_ref, mel_deg, data_range=mel_ref.max() - mel_ref.min())
        return float(score)
    except ImportError:
        print("    [WARN] scikit-image not installed, SSIM skipped. pip install scikit-image")
        return None
    except Exception as e:
        print(f"    [WARN] SSIM error: {e}")
        return None


def evaluate_single(ref_path, gen_path, sr=22050):
    """Evaluate all metrics for one file pair."""
    ref_wav = load_audio(ref_path, sr=sr)
    gen_wav = load_audio(gen_path, sr=sr)

    return {
        "reference": os.path.basename(ref_path),
        "generated": os.path.basename(gen_path),
        "pesq": compute_pesq(ref_wav, gen_wav, sr),
        "stoi": compute_stoi(ref_wav, gen_wav, sr),
        "ssim_mel": compute_ssim_mel(ref_wav, gen_wav, sr),
    }


def evaluate_directory(gen_dir, ref_dir=None, results_file=None):
    """Evaluate all wav files in directory."""
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.wav")))
    if not gen_files:
        print(f"[ERROR] No .wav files in {gen_dir}")
        return []

    print(f"\nFound {len(gen_files)} files in {gen_dir}")
    if ref_dir:
        print(f"Reference dir: {ref_dir}")

    all_metrics = []

    for i, gen_path in enumerate(gen_files):
        filename = os.path.basename(gen_path)
        print(f"  [{i+1}/{len(gen_files)}] {filename}")

        metrics = {"generated": filename, "pesq": None, "stoi": None, "ssim_mel": None}

        if ref_dir:
            ref_path = os.path.join(ref_dir, filename)
            if os.path.exists(ref_path):
                result = evaluate_single(ref_path, gen_path)
                metrics.update(result)
            else:
                print(f"    [WARN] Reference not found: {filename}, skipping PESQ/STOI/SSIM")

        all_metrics.append(metrics)

        parts = []
        if metrics.get("pesq") is not None:
            parts.append(f"PESQ={metrics['pesq']:.3f}")
        if metrics.get("stoi") is not None:
            parts.append(f"STOI={metrics['stoi']:.3f}")
        if metrics.get("ssim_mel") is not None:
            parts.append(f"SSIM={metrics['ssim_mel']:.3f}")
        if parts:
            print(f"    {' | '.join(parts)}")

    # Summary
    print_summary(all_metrics, os.path.basename(gen_dir))

    # Save
    if results_file:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        report = {
            "model": os.path.basename(gen_dir),
            "num_files": len(all_metrics),
            "summary": get_summary(all_metrics),
            "per_file": all_metrics,
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"\nResults saved: {results_file}")

    return all_metrics


def get_summary(metrics_list):
    """Compute average metrics."""
    summary = {}
    for key in ["pesq", "stoi", "ssim_mel"]:
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        if vals:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return summary


def print_summary(metrics_list, model_name=""):
    """Print summary table."""
    summary = get_summary(metrics_list)

    print(f"\n{'=' * 55}")
    print(f" SUMMARY — {model_name}")
    print(f"{'=' * 55}")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    labels = {"pesq": "PESQ", "stoi": "STOI", "ssim_mel": "SSIM"}
    for key, label in labels.items():
        vals = summary.get(key, {})
        if vals:
            print(f"{label:<12} {vals['mean']:>8.4f} {vals['std']:>8.4f} {vals['min']:>8.4f} {vals['max']:>8.4f}")
    print(f"{'=' * 55}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech quality")
    parser.add_argument("--gen_dir", type=str, help="Directory with generated wav files")
    parser.add_argument("--ref_dir", type=str, default=None, help="Directory with reference wav files")
    parser.add_argument("--output", type=str, default="output/metrics/metrics_report.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    if not args.gen_dir:
        print("Usage:")
        print("  python evaluate_metrics.py --gen_dir output/synthesis/tacotron2 --ref_dir data/LJSpeech-1.1/wavs")
        print("  python evaluate_metrics.py --gen_dir output/synthesis/tacotron2  (no reference)")
        sys.exit(1)

    print("=" * 55)
    print("Speech Quality Evaluation — PESQ, STOI, SSIM")
    print("=" * 55)

    evaluate_directory(args.gen_dir, args.ref_dir, args.output)


if __name__ == "__main__":
    main()
