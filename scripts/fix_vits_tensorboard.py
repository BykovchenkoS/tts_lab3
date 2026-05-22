import os
import sys
import re
import glob
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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


def parse_log(filepath):
    """Parse VITS trainer_0_log.txt into step-wise metrics."""
    data = []
    current_step = None
    current_metrics = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Match GLOBAL_STEP line
            gs_match = re.search(r"GLOBAL_STEP:\s*(\d+)", line)
            if gs_match:
                if current_step is not None and current_metrics:
                    current_metrics["__step__"] = current_step
                    data.append(current_metrics)

                current_step = int(gs_match.group(1))
                current_metrics = {}
                continue

            if current_step is not None and "| > " in line:
                m = re.match(r"\s*\|\s*>\s*(\S+):\s*(.*?)\s*\(", line.strip())
                if not m:
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

    if current_step is not None and current_metrics:
        current_metrics["__step__"] = current_step
        data.append(current_metrics)

    return data


def main():
    parser = argparse.ArgumentParser(description="Fix VITS TensorBoard events")
    parser.add_argument("--log_dir", default="run/vits",
                        help="Path to VITS run directory")
    parser.add_argument("--steps_per_epoch", type=int, default=205,
                        help="Steps per epoch")
    args = parser.parse_args()

    log_files = glob.glob(os.path.join(args.log_dir, "*", "trainer_0_log.txt"))
    if not log_files:
        print(f"[ERROR] No trainer_0_log.txt found in {args.log_dir}")
        sys.exit(1)

    log_file = log_files[0]
    print(f"Log file: {log_file}")

    print("Parsing log...")
    data = parse_log(log_file)
    print(f"Found {len(data)} logged steps")

    if not data:
        print("[ERROR] No data parsed from log")
        sys.exit(1)

    tb_dir = os.path.join(args.log_dir, "tb_fixed")
    writer = SummaryWriter(log_dir=tb_dir)

    key_metrics = {
        "loss_1": "Loss/Generator Total",
        "loss_disc": "Loss/Discriminator",
        "loss_gen": "Loss/Generator Adversarial",
        "loss_mel": "Loss/Mel Spectrogram",
        "loss_kl": "Loss/KL Divergence",
        "loss_feat": "Loss/Feature Matching",
        "loss_duration": "Loss/Duration",
        "grad_norm_0": "GradNorm/Discriminator",
        "grad_norm_1": "GradNorm/Generator",
        "current_lr_0": "LR/Discriminator",
        "current_lr_1": "LR/Generator",
    }

    all_metrics = set()
    for entry in data:
        for k in entry:
            if k != "__step__":
                all_metrics.add(k)

    written = 0
    for entry in data:
        step = entry.get("__step__", 0)
        epoch = step / args.steps_per_epoch

        for key, tag in key_metrics.items():
            if key in entry:
                writer.add_scalar(tag, entry[key], global_step=epoch)

        written += 1

    writer.close()
    print(f"Wrote {written} data points to {tb_dir}")
    print(f"\nNow run TensorBoard:")
    print(f"  tensorboard --logdir {tb_dir}")
    print(f"\nOr to see both Tacotron 2 and VITS:")
    print(f"  tensorboard --logdir run")


if __name__ == "__main__":
    main()
