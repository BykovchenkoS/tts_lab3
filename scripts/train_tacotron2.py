import os
import json
import subprocess
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Train Tacotron 2")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--small", type=int, default=None,
                        help="Use only N samples for fast debug (e.g. 64)")
    parser.add_argument("--continue_path", type=str, default=None,
                        help="Path to previous run to resume training from")
    args = parser.parse_args()

    # Paths
    config_path = os.path.join(PROJECT_ROOT, "config", "tacotron2_run_config.json")
    dataset_path = os.path.join(PROJECT_ROOT, "data", "raw", "LJSpeech-1.1")
    output_path = os.path.join(PROJECT_ROOT, "run", "tacotron2")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Check dataset
    if not os.path.isdir(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("        Run  python scripts/prepare_data.py  first!")
        return 1

    # ---- Generate config ----
    config = {
        "run_name": "tacotron2_ljspeech",
        "model": "tacotron2",
        "output_path": output_path,

        # ---- Audio processing ----
        "audio": {
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "num_mels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "signal_norm": True,
            "symmetric_norm": True,
            "max_norm": 4.0,
            "clip_norm": True,
        },

        # ---- Dataset (formatter="ljspeech" for Coqui TTS) ----
        "datasets": [{
            "formatter": "ljspeech",
            "path": dataset_path,
            "meta_file_train": "metadata.csv",
            "meta_file_val": "metadata.csv",
            "language": "en",
            "phonemizer": "espeak",
        }],

        # ---- Dataloader ----
        "batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 0,

        # ---- Training schedule ----
        "epochs": args.epochs,
        "test_eval_epochs": 10,
        "print_step": 25,
        "print_eval": True,
        "save_step": max(1, args.epochs // 20),
        "checkpoint": True,

        # ---- Early stopping ----
        "eval_avg_loss_epochs": 2,
        "early_stop_patience": 50,
        "early_stop_min_delta": 0.05,

        # ---- Optimizer ----
        "optimizer": "Adam",
        "optimizer_params": {"betas": [0.9, 0.998], "weight_decay": 1e-6},
        "lr": 1e-3,
        "lr_scheduler": "NoamLR",
        "lr_scheduler_params": {"warmup_steps": 4000},

        # ---- Regularisation ----
        "grad_clip": 1.0,
        "cudnn_benchmark": False,
        "seed": 54321,

        # ---- Model specifics ----
        "use_spectral_norm": False,
        "stopnet_weight_decay": 0.0,
        "enable_eos_bos_chars": True,

        # ---- Logging (TensorBoard) ----
        "logger": "tensorboard",
        "tb_model_param_stats": True,
        "tb_log_step": 50,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---- Header ----
    print("=" * 60)
    print("  Tacotron 2 Training  —  Coqui TTS")
    print("=" * 60)
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch_size}")
    if args.small:
        print(f"  Small  : {args.small} samples (debug mode)")
    print(f"  Config : {config_path}")
    print(f"  Output : {output_path}")
    print(f"  Early stop: patience={config['early_stop_patience']} epochs")
    print(f"  TensorBoard: tensorboard --logdir {output_path}")
    print("=" * 60)

    # ---- Launch training ----
    cmd = [
        sys.executable, "-m", "TTS.bin.train_tts",
        "--config_path", config_path,
    ]

    if args.small:
        cmd.extend(["--small_run", str(args.small)])

    if args.continue_path:
        cmd.extend(["--continue_path", args.continue_path])
        print(f"  Resuming from: {args.continue_path}")

    print(f"\n  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print("\n  [OK] Training complete!")
    else:
        print(f"\n  [EXIT] Code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
