import os
import sys
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_checkpoint(output_dir):
    """Find the latest checkpoint (best_model.pth or checkpoint.pth)."""
    candidates = [
        os.path.join(output_dir, "best_model.pth"),
        os.path.join(output_dir, "checkpoint.pth"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Search subdirectories
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f == "best_model.pth":
                return os.path.join(root, f)
            if f == "checkpoint.pth":
                return os.path.join(root, f)
    return None


def find_config(config_dir, model_name):
    """Find config file for model."""
    candidates = [
        os.path.join(config_dir, f"{model_name}_run_config.json"),
        os.path.join(config_dir, f"{model_name}_config.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Try to find any config with model name
    if os.path.isdir(config_dir):
        for f in os.listdir(config_dir):
            if model_name in f.lower() and f.endswith(".json"):
                return os.path.join(config_dir, f)
    return None


def synthesize_python_api(model_name, config_path, checkpoint_path,
                          output_dir, text_file):
    """Synthesize using TTS Python API (most reliable method)."""
    try:
        from TTS.api import TTS

        tts = TTS(model_path=checkpoint_path, config_path=config_path)

        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for i, line in enumerate(lines):
            out_file = os.path.join(output_dir, f"{model_name}_{i:03d}.wav")
            print(f"  [{i+1}/{len(lines)}] {line[:60]}...")
            tts.tts_to_file(text=line, file_path=out_file)

        return 0

    except ImportError:
        print("  [WARN] TTS.api not available, trying alternative...")
        return 1
    except Exception as e:
        print(f"  [WARN] TTS.api failed: {e}")
        return 1


def synthesize_python_direct(model_name, config_path, checkpoint_path,
                             output_dir, text_file):
    """Synthesize using direct model loading."""
    try:
        from TTS.config import load_config
        from TTS.tts.models import setup_model
        import torch
        import soundfile as sf

        config = load_config(config_path)
        model = setup_model(config)
        model.load_checkpoint(config, checkpoint_path, eval=True)

        if torch.cuda.is_available():
            model.cuda()

        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for i, line in enumerate(lines):
            print(f"  [{i+1}/{len(lines)}] {line[:60]}...")
            with torch.no_grad():
                wav = model.synthesis(line)
            out_file = os.path.join(output_dir, f"{model_name}_{i:03d}.wav")
            sf.write(out_file, wav["wav"], config.audio["sample_rate"])

        return 0

    except Exception as e:
        print(f"  [WARN] Direct model loading failed: {e}")
        return 1


def synthesize_model(model_name):
    """Synthesize speech with a specific model."""
    print(f"\n{'=' * 60}")
    print(f"  Synthesizing: {model_name.upper()}")
    print(f"{'=' * 60}")

    output_model_dir = os.path.join(PROJECT_ROOT, "run", model_name)
    config_dir = os.path.join(PROJECT_ROOT, "config")
    synth_dir = os.path.join(PROJECT_ROOT, "output", "synthesis", model_name)
    text_file = os.path.join(PROJECT_ROOT, "tts_test_text.txt")

    os.makedirs(synth_dir, exist_ok=True)

    # Find config
    config_path = find_config(config_dir, model_name)
    if config_path is None:
        print(f"[ERROR] Config not found for {model_name} in {config_dir}")
        return 1

    # Find checkpoint
    checkpoint = find_checkpoint(output_model_dir)
    if checkpoint is None:
        print(f"[ERROR] No checkpoint found in {output_model_dir}")
        print("        Train the model first!")
        return 1

    print(f"  Config    : {config_path}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Text file : {text_file}")
    print(f"  Output dir: {synth_dir}")

    # Try synthesis methods in order
    methods = [
        ("TTS API", lambda: synthesize_python_api(
            model_name, config_path, checkpoint, synth_dir, text_file)),
        ("Direct model", lambda: synthesize_python_direct(
            model_name, config_path, checkpoint, synth_dir, text_file)),
    ]

    for method_name, method_fn in methods:
        print(f"\n  [INFO] Trying {method_name}...")
        try:
            rc = method_fn()
            if rc == 0:
                print(f"\n  [OK] Synthesis complete via {method_name}!")
                return 0
        except Exception as e:
            print(f"  [WARN] {method_name} failed: {e}")

    print(f"\n  [ERROR] All synthesis methods failed for {model_name}")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech")
    parser.add_argument("--model", choices=["tacotron2", "vits", "both"],
                        default="both")
    parser.add_argument("--text", type=str, default=None,
                        help="Custom text (overrides tts_test_text.txt)")
    args = parser.parse_args()

    # Write custom text if provided
    if args.text is not None:
        text_file = os.path.join(PROJECT_ROOT, "tts_test_text.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(args.text)
        print(f"[INFO] Using custom text -> {text_file}")

    models = []
    if args.model in ("tacotron2", "both"):
        models.append("tacotron2")
    if args.model in ("vits", "both"):
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