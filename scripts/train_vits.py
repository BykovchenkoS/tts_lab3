import os
import subprocess
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Train VITS")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--small", type=int, default=None,
                        help="Use only N samples for fast debug (e.g. 64)")
    parser.add_argument("--continue_path", type=str, default=None,
                        help="Path to previous run to resume training from")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, "config", "vits_run_config.json")
    default_config = os.path.join(PROJECT_ROOT, "config", "vits_default.json")
    dataset_path = os.path.join(PROJECT_ROOT, "data", "raw", "LJSpeech-1.1")
    output_path = os.path.join(PROJECT_ROOT, "run", "vits")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("        Run  python scripts/prepare_data.py  first!")
        return 1

    if not os.path.isfile(default_config):
        print("[INFO] Generating default VITS config...")
        from TTS.tts.configs.vits_config import VitsConfig
        VitsConfig().save_json(default_config)

    from TTS.config import load_config

    config = load_config(default_config)

    config.run_name = "vits_ljspeech"
    config.output_path = output_path

    config.datasets = [{
        "formatter": "ljspeech",
        "path": dataset_path,
        "meta_file_train": "metadata.csv",
        "meta_file_val": "metadata.csv",
        "language": "en",
        "phonemizer": "espeak",
    }]

    config.batch_size = args.batch_size
    config.eval_batch_size = args.batch_size
    config.num_loader_workers = 4
    config.num_eval_loader_workers = 0

    config.epochs = args.epochs
    config.test_eval_epochs = 10
    config.print_step = 25
    config.print_eval = True
    config.save_step = max(1, args.epochs // 10)
    config.checkpoint = True

    try:
        config.eval_avg_loss_epochs = 2
        config.early_stop_patience = 50
        config.early_stop_min_delta = 0.05
    except AttributeError:
        pass

    config.optimizer = "AdamW"
    config.lr = 2e-4
    step_size = max(1, args.epochs // 3)
    config.lr_scheduler = "StepLR"
    config.lr_scheduler_params = {"step_size": step_size, "gamma": 0.5}

    config.lr_disc = 2e-4
    config.lr_scheduler_disc = "StepLR"
    config.lr_scheduler_disc_params = {"step_size": step_size, "gamma": 0.5}

    config.cudnn_benchmark = False
    config.seed = 54321

    config.use_speaker_embedding = False

    config.logger = "tensorboard"
    config.tb_log_step = 50
    try:
        config.tb_model_param_stats = True
    except AttributeError:
        pass

    config.save_json(config_path)

    print("=" * 60)
    print("  VITS Training  -  Coqui TTS")
    print("=" * 60)
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch_size}")
    if args.small:
        print(f"  Small  : {args.small} samples (debug mode)")
    print(f"  Config : {config_path}")
    print(f"  Output : {output_path}")
    print(f"  TensorBoard: tensorboard --logdir {output_path}")
    print("=" * 60)

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
