import os
import sys
import glob
import random
import argparse
import json
import subprocess
import numpy as np

import librosa


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_audio(filepath, sr=22050):
    return librosa.load(filepath, sr=sr)[0]


def resample_wav(wav, orig_sr, target_sr):
    return librosa.resample(y=wav, orig_sr=orig_sr, target_sr=target_sr)


def align_length(ref, deg):
    min_len = min(len(ref), len(deg))
    return ref[:min_len], deg[:min_len]


def compute_pesq(ref, deg, sr=22050):
    try:
        from pesq import pesq as pesq_fn
        if sr != 16000:
            ref = resample_wav(ref, sr, 16000)
            deg = resample_wav(deg, sr, 16000)
        ref, deg = align_length(ref, deg)
        return float(pesq_fn(16000, ref, deg, "wb"))
    except Exception as e:
        print(f"    [WARN] PESQ: {e}")
        return None


def compute_stoi(ref, deg, sr=22050):
    try:
        from pystoi.stoi import stoi as stoi_fn
        if sr != 16000:
            ref = resample_wav(ref, sr, 16000)
            deg = resample_wav(deg, sr, 16000)
        ref, deg = align_length(ref, deg)
        return float(stoi_fn(ref, deg, 16000, extended=False))
    except Exception as e:
        print(f"    [WARN] STOI: {e}")
        return None


def compute_ssim_mel(ref, deg, sr=22050):
    try:
        from skimage.metrics import structural_similarity as ssim
        S_ref = librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=80, n_fft=1024, hop_length=256)
        S_deg = librosa.feature.melspectrogram(y=deg, sr=sr, n_mels=80, n_fft=1024, hop_length=256)
        min_t = min(S_ref.shape[1], S_deg.shape[1])
        S_ref, S_deg = S_ref[:, :min_t], S_deg[:, :min_t]
        return float(ssim(S_ref, S_deg, data_range=S_ref.max() - S_ref.min()))
    except Exception as e:
        print(f"    [WARN] SSIM: {e}")
        return None


def find_ljspeech_metadata(ljspeech_dir):
    meta_path = os.path.join(ljspeech_dir, "metadata.csv")
    if not os.path.isfile(meta_path):
        print(f"[ERROR] metadata.csv not found at {meta_path}")
        return []
    entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                file_id = parts[0].strip()
                text = parts[1].strip()
                wav_path = os.path.join(ljspeech_dir, "wavs", file_id + ".wav")
                if os.path.isfile(wav_path):
                    entries.append({"id": file_id, "text": text, "wav": wav_path})
    return entries


def synthesize_one(config_path, checkpoint_path, text, output_path):
    cmd = [
        sys.executable, "-m", "TTS.bin.synthesize",
        "--model_path", checkpoint_path,
        "--config_path", config_path,
        "--text", text,
        "--out_path", output_path,
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=600,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ERROR] Synthesis failed: {result.stderr[:200]}")
        return False
    return os.path.isfile(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_run", required=True, help="Path to model run directory")
    parser.add_argument("--ljspeech_dir", default=os.path.join(PROJECT_ROOT, "data", "LJSpeech-1.1"),
                        help="Path to LJSpeech-1.1")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "output", "eval_pairs"),
                        help="Output directory for synthesized files and results")
    parser.add_argument("--results_file", default=None, help="Path to save JSON results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = os.path.join(args.model_run, "best_model.pth")
    config = os.path.join(args.model_run, "config.json")
    if not os.path.isfile(checkpoint):
        checkpoints = sorted(glob.glob(os.path.join(args.model_run, "checkpoint_*.pth")))
        if checkpoints:
            checkpoint = checkpoints[-1]
    if not os.path.isfile(checkpoint):
        print(f"[ERROR] No checkpoint found in {args.model_run}")
        sys.exit(1)
    if not os.path.isfile(config):
        print(f"[ERROR] No config.json in {args.model_run}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint}")
    print(f"Config:     {config}")

    entries = find_ljspeech_metadata(args.ljspeech_dir)
    if not entries:
        print("[ERROR] No LJSpeech entries found!")
        sys.exit(1)
    print(f"LJSpeech:   {len(entries)} entries")

    random.seed(42)
    samples = random.sample(entries, min(args.n_samples, len(entries)))
    print(f"Selected:   {len(samples)} samples\n")

    results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['id']}: {sample['text'][:60]}...")

        synth_path = os.path.join(args.output_dir, f"synth_{sample['id']}.wav")
        ok = synthesize_one(config, checkpoint, sample["text"], synth_path)
        if not ok:
            results.append({"id": sample["id"], "text": sample["text"],
                            "pesq": None, "stoi": None, "ssim_mel": None})
            continue

        ref_wav = load_audio(sample["wav"])
        synth_wav = load_audio(synth_path)

        pesq = compute_pesq(ref_wav, synth_wav)
        stoi = compute_stoi(ref_wav, synth_wav)
        ssim = compute_ssim_mel(ref_wav, synth_wav)

        parts = []
        if pesq is not None: parts.append(f"PESQ={pesq:.3f}")
        if stoi is not None: parts.append(f"STOI={stoi:.3f}")
        if ssim is not None: parts.append(f"SSIM={ssim:.3f}")
        print(f"    {' | '.join(parts)}")

        results.append({
            "id": sample["id"],
            "text": sample["text"],
            "reference": sample["wav"],
            "synthesized": synth_path,
            "pesq": pesq,
            "stoi": stoi,
            "ssim_mel": ssim,
        })

    print(f"\n{'=' * 60}")
    print(f" SUMMARY — Tacotron 2 vs LJSpeech Reference")
    print(f"{'=' * 60}")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for key, label in [("pesq", "PESQ"), ("stoi", "STOI"), ("ssim_mel", "SSIM")]:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(f"{label:<12} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}")
    print(f"{'=' * 60}")

    print("\nInterpretation:")
    pesq_vals = [r["pesq"] for r in results if r.get("pesq") is not None]
    stoi_vals = [r["stoi"] for r in results if r.get("stoi") is not None]
    if pesq_vals:
        avg = np.mean(pesq_vals)
        if avg >= 3.5:
            print(f"  PESQ = {avg:.2f} — Excellent")
        elif avg >= 2.5:
            print(f"  PESQ = {avg:.2f} — Good")
        elif avg >= 1.5:
            print(f"  PESQ = {avg:.2f} — Fair")
        else:
            print(f"  PESQ = {avg:.2f} — Poor")
    if stoi_vals:
        avg = np.mean(stoi_vals)
        if avg >= 0.9:
            print(f"  STOI = {avg:.2f} — Excellent intelligibility")
        elif avg >= 0.75:
            print(f"  STOI = {avg:.2f} — Good intelligibility")
        elif avg >= 0.6:
            print(f"  STOI = {avg:.2f} — Fair intelligibility")
        else:
            print(f"  STOI = {avg:.2f} — Poor intelligibility")

    results_file = args.results_file or os.path.join(args.output_dir, "eval_results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    main()
