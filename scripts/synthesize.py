import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_checkpoint(output_dir):
    for root, dirs, files in os.walk(output_dir):
        dirs.sort(reverse=True)
        for f in files:
            if f == "best_model.pth":
                return os.path.join(root, f)
    return None


def find_config_for_checkpoint(checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_in_run = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(config_in_run):
        return config_in_run
    return None


def synthesize_one(model_name, config_path, checkpoint_path, text, output_file):
    cmd = [
        sys.executable, "-m", "TTS.bin.synthesize",
        "--model_path", checkpoint_path,
        "--config_path", config_path,
        "--text", text,
        "--out_path", output_file,
    ]
    print(f"    -> {output_file}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=600)
    return result.returncode


def synthesize_model(model_name):
    print(f"\n{'=' * 60}")
    print(f"  Synthesizing: {model_name.upper()}")
    print(f"{'=' * 60}")

    output_model_dir = os.path.join(PROJECT_ROOT, "run", model_name)
    synth_dir = os.path.join(PROJECT_ROOT, "output", "synthesis", model_name)
    text_file = os.path.join(PROJECT_ROOT, "tts_test_text.txt")

    os.makedirs(synth_dir, exist_ok=True)

    checkpoint = find_checkpoint(output_model_dir)
    if checkpoint is None:
        print(f"[ERROR] No best_model.pth found in {output_model_dir}")
        return 1

    config_path = find_config_for_checkpoint(checkpoint)
    if config_path is None:
        print(f"[ERROR] No config.json found next to checkpoint: {checkpoint}")
        return 1

    if not os.path.isfile(text_file):
        print(f"[ERROR] Text file not found: {text_file}")
        return 1

    with open(text_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    print(f"  Checkpoint: {checkpoint}")
    print(f"  Config    : {config_path}")
    print(f"  Text file : {text_file} ({len(lines)} lines)")
    print(f"  Output dir: {synth_dir}")
    print()

    success = 0
    failed = 0
    for i, line in enumerate(lines):
        out_file = os.path.join(synth_dir, f"{model_name}_{i:03d}.wav")
        print(f"  [{i+1}/{len(lines)}] {line[:70]}...")
        rc = synthesize_one(model_name, config_path, checkpoint, line, out_file)
        if rc == 0:
            success += 1
        else:
            failed += 1

    print(f"\n  Result: {success} OK, {failed} failed")
    return 0 if failed == 0 else 1


def main():
    model = "both"
    custom_text = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--text" and i + 1 < len(args):
            custom_text = " ".join(args[i + 1:])
            i += 2
        else:
            i += 1

    if model not in ("tacotron2", "vits", "both"):
        print(f"[ERROR] --model must be tacotron2, vits, or both (got: {model})")
        return 1

    if custom_text is not None:
        text_file = os.path.join(PROJECT_ROOT, "tts_test_text.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(custom_text + "\n")
        print(f"[INFO] Using custom text -> {text_file}")

    models = []
    if model in ("tacotron2", "both"):
        models.append("tacotron2")
    if model in ("vits", "both"):
        models.append("vits")

    results = {}
    for m in models:
        results[m] = synthesize_model(m)

    print(f"\n{'=' * 60}")
    print("  Synthesis Summary")
    print(f"{'=' * 60}")
    for m, rc in results.items():
        status = "OK" if rc == 0 else "FAILED"
        print(f"  {m:15s} : {status}")
    print(f"{'=' * 60}\n")

    return max(results.values())


if __name__ == "__main__":
    sys.exit(main())
