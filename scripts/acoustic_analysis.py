import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.audio_utils import load_audio, mel_spectrogram, compute_spectral_features


def extract_features_from_directory(
    audio_dir: str,
    sr: int = 22050,
    max_files: int = None,
) -> dict:
    """
    Извлечь акустические признаки из всех wav файлов в директории.
    Возвращает словарь с усреднёнными значениями по всем файлам.
    """
    import glob

    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_files:
        print(f"[WARN] Нет .wav файлов в {audio_dir}")
        return {}

    if max_files:
        wav_files = wav_files[:max_files]

    print(f"[INFO] Извлечение признаков из {len(wav_files)} файлов: {audio_dir}")

    # Накопители
    all_features = defaultdict(list)
    mel_spectrograms = []

    for i, wav_path in enumerate(wav_files):
        print(f"  [{i + 1}/{len(wav_files)}] {os.path.basename(wav_path)}")

        wav = load_audio(wav_path, sr=sr)
        features = compute_spectral_features(wav, sr=sr)
        mel = mel_spectrogram(wav, sr=sr)

        # Накапливаем средние значения для каждого признака
        for key, value in features.items():
            if value is not None:
                # Для 1D признаков берём среднее
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    valid = value[~np.isnan(value)]
                    if len(valid) > 0:
                        all_features[key].append(np.mean(valid))
                        all_features[f"{key}_std"].append(np.std(valid))
                # Для 2D признаков (spectral_contrast) берём среднее по всему
                elif isinstance(value, np.ndarray) and value.ndim == 2:
                    valid = value[~np.isnan(value)]
                    if len(valid) > 0:
                        all_features[key].append(np.mean(valid))
                        all_features[f"{key}_std"].append(np.std(valid))

        mel_spectrograms.append(mel)

    # Агрегируем: среднее и std по файлам
    aggregated = {}
    for key, values in all_features.items():
        if len(values) > 0:
            valid = [v for v in values if not np.isnan(v)]
            if valid:
                aggregated[key] = {
                    "mean": float(np.mean(valid)),
                    "std": float(np.std(valid)),
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                }

    aggregated["num_files"] = len(wav_files)
    aggregated["mel_spectrograms"] = mel_spectrograms

    return aggregated


def compute_statistics_comparison(features_dict: dict, name: str) -> dict:
    """Форматировать признаки для сравнительной таблицы."""
    stats = {"name": name}
    keys_of_interest = [
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "spectral_flatness", "zero_crossing_rate", "rms",
        "f0",
    ]

    for key in keys_of_interest:
        if key in features_dict:
            stats[key] = features_dict[key]
        # MFCC
        for i in range(1, 14):
            mfcc_key = f"mfcc_{i}"
            if mfcc_key in features_dict:
                stats[mfcc_key] = features_dict[mfcc_key]

    return stats


def save_comparison_table(
    features_list: list[dict],
    output_path: str,
):
    """Сохранить сравнительную таблицу в JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Сравнительная таблица сохранена: {output_path}")


def plot_comparison(
    features_list: list[tuple[str, dict]],
    output_dir: str,
):
    """
    Построить графики сравнения акустических признаков.
    features_list: [(name, features_dict), ...]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Настройка шрифтов для русского (если нужно)
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf")
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    except Exception:
        pass

    os.makedirs(output_dir, exist_ok=True)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # === 1. Сравнение MFCC (bar chart) ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    mfcc_means = {}
    for name, feats in features_list:
        means = [feats.get(f"mfcc_{i}", {}).get("mean", 0) for i in range(1, 14)]
        mfcc_means[name] = means

    x = np.arange(13)
    width = 0.25
    for idx, (name, means) in enumerate(mfcc_means.items()):
        offset = (idx - len(mfcc_means) / 2 + 0.5) * width
        axes[0].bar(x + offset, means, width, label=name, color=colors[idx % len(colors)], alpha=0.8)
    axes[0].set_xlabel("MFCC Coefficient")
    axes[0].set_ylabel("Mean Value")
    axes[0].set_title("MFCC Mean Values Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(i) for i in range(1, 14)])
    axes[0].legend(loc="best")
    axes[0].grid(axis="y", alpha=0.3)

    # MFCC Std
    mfcc_stds = {}
    for name, feats in features_list:
        stds = [feats.get(f"mfcc_{i}", {}).get("std", 0) for i in range(1, 14)]
        mfcc_stds[name] = stds

    for idx, (name, stds) in enumerate(mfcc_stds.items()):
        offset = (idx - len(mfcc_stds) / 2 + 0.5) * width
        axes[1].bar(x + offset, stds, width, label=name, color=colors[idx % len(colors)], alpha=0.8)
    axes[1].set_xlabel("MFCC Coefficient")
    axes[1].set_ylabel("Std Value")
    axes[1].set_title("MFCC Std Values Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i) for i in range(1, 14)])
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mfcc_comparison.png"), dpi=150)
    plt.close()
    print(f"[INFO] Сохранён: {output_dir}/mfcc_comparison.png")

    # === 2. Спектральные признаки (grouped bar) ===
    spectral_keys = [
        ("spectral_centroid", "Spectral Centroid (Hz)"),
        ("spectral_bandwidth", "Spectral Bandwidth (Hz)"),
        ("spectral_rolloff", "Spectral Rolloff (Hz)"),
        ("zero_crossing_rate", "Zero Crossing Rate"),
        ("rms", "RMS Energy"),
        ("spectral_flatness", "Spectral Flatness"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(spectral_keys):
        means = []
        stds = []
        names = []
        for name, feats in features_list:
            if key in feats:
                means.append(feats[key]["mean"])
                stds.append(feats[key]["std"])
                names.append(name)

        if means:
            x_pos = np.arange(len(names))
            bars = axes[idx].bar(x_pos, means, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.8, yerr=stds, capsize=3)
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(names, fontsize=9)
        axes[idx].set_title(title, fontsize=11)
        axes[idx].grid(axis="y", alpha=0.3)

    plt.suptitle("Spectral Features Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectral_comparison.png"), dpi=150)
    plt.close()
    print(f"[INFO] Сохранён: {output_dir}/spectral_comparison.png")

    # === 3. F0 распределение ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F0 mean
    f0_names = []
    f0_means = []
    f0_stds = []
    for name, feats in features_list:
        if "f0" in feats:
            f0_names.append(name)
            f0_means.append(feats["f0"]["mean"])
            f0_stds.append(feats["f0"]["std"])

    if f0_means:
        axes[0].bar(f0_names, f0_means, color=[colors[i % len(colors)] for i in range(len(f0_names))], alpha=0.8, yerr=f0_stds, capsize=3)
    axes[0].set_title("F0 (Pitch) Mean Comparison")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].grid(axis="y", alpha=0.3)

    # F0 range
    f0_ranges = []
    for name, feats in features_list:
        if "f0" in feats:
            f0_ranges.append(feats["f0"]["max"] - feats["f0"]["min"])

    if f0_ranges:
        axes[1].bar(f0_names, f0_ranges, color=[colors[i % len(colors)] for i in range(len(f0_names))], alpha=0.8)
    axes[1].set_title("F0 (Pitch) Range Comparison")
    axes[1].set_ylabel("Range (Hz)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f0_comparison.png"), dpi=150)
    plt.close()
    print(f"[INFO] Сохранён: {output_dir}/f0_comparison.png")

    # === 4. Мел-спектрограммы (образцы) ===
    fig, axes = plt.subplots(len(features_list), 1, figsize=(14, 4 * len(features_list)))

    for idx, (name, feats) in enumerate(features_list):
        mels = feats.get("mel_spectrograms", [])
        if mels:
            # Показываем первую мел-спектрограмму как образец
            ax = axes[idx] if len(features_list) > 1 else axes
            im = ax.imshow(mels[0], aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"Mel Spectrogram — {name}")
            ax.set_xlabel("Time Frame")
            ax.set_ylabel("Mel Bin")
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mel_spectrograms_comparison.png"), dpi=150)
    plt.close()
    print(f"[INFO] Сохранён: {output_dir}/mel_spectrograms_comparison.png")

    # === 5. Radar chart (MFCC средние, первые 5 коэффициентов) ===
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = [f"MFCC-{i}" for i in range(1, 6)]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for idx, (name, feats) in enumerate(features_list):
        values = [feats.get(f"mfcc_{i}", {}).get("mean", 0) for i in range(1, 6)]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("MFCC Radar (Coefficients 1-5)", y=1.08)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mfcc_radar.png"), dpi=150)
    plt.close()
    print(f"[INFO] Сохранён: {output_dir}/mfcc_radar.png")


def main():
    parser = argparse.ArgumentParser(description="Сравнительный анализ акустических признаков")

    parser.add_argument("--ref", type=str, default=None, help="Директория с референсными аудио (LJSpeech)")
    parser.add_argument("--gen", type=str, default=None, help="Директория с сгенерированными аудио (одна модель)")
    parser.add_argument("--gen_t2", type=str, default=None, help="Директория с аудио Tacotron 2")
    parser.add_argument("--gen_vits", type=str, default=None, help="Директория с аудио VITS")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "generated", "analysis"),
                        help="Директория для графиков")
    parser.add_argument("--output_json", type=str, default=None, help="Путь к JSON с результатами")
    parser.add_argument("--max_files", type=int, default=None, help="Макс. файлов для анализа (для ускорения)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")

    args = parser.parse_args()

    print("=" * 60)
    print("Сравнительный анализ акустических признаков")
    print("=" * 60)

    features_list = []

    # 1. Референс
    if args.ref:
        ref_feats = extract_features_from_directory(args.ref, sr=args.sr, max_files=args.max_files)
        if ref_feats:
            stats = compute_statistics_comparison(ref_feats, "Reference (LJSpeech)")
            features_list.append(("Reference (LJSpeech)", stats))

    # 2. Одна модель
    if args.gen:
        gen_feats = extract_features_from_directory(args.gen, sr=args.sr, max_files=args.max_files)
        if gen_feats:
            name = os.path.basename(args.gen)
            stats = compute_statistics_comparison(gen_feats, name)
            features_list.append((name, stats))

    # 3. Две модели
    if args.gen_t2:
        t2_feats = extract_features_from_directory(args.gen_t2, sr=args.sr, max_files=args.max_files)
        if t2_feats:
            stats = compute_statistics_comparison(t2_feats, "Tacotron 2")
            features_list.append(("Tacotron 2", stats))

    if args.gen_vits:
        vits_feats = extract_features_from_directory(args.gen_vits, sr=args.sr, max_files=args.max_files)
        if vits_feats:
            stats = compute_statistics_comparison(vits_feats, "VITS")
            features_list.append(("VITS", stats))

    if not features_list:
        print("[ERROR] Не указаны директории с аудио!")
        print("  Пример: --ref data/raw/LJSpeech-1.1/wavs --gen data/generated/tacotron2")
        return

    # 4. Вывод сравнительной таблицы
    print(f"\n{'=' * 80}")
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА АКУСТИЧЕСКИХ ПРИЗНАКОВ")
    print(f"{'=' * 80}")

    feature_keys = [
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "spectral_flatness", "zero_crossing_rate", "rms", "f0",
    ]

    # Заголовок
    header = f"{'Признак':<25}"
    for name, _ in features_list:
        header += f" | {name:>20}"
    print(header)
    print("-" * (25 + 25 * len(features_list)))

    for key in feature_keys:
        row = f"{key:<25}"
        for _, feats in features_list:
            if key in feats:
                mean = feats[key]["mean"]
                std = feats[key]["std"]
                row += f" | {mean:>10.2f} ± {std:<8.2f}"
            else:
                row += f" | {'N/A':>20}"
        print(row)

    # MFCC
    print(f"\n{'MFCC':<25}")
    for i in range(1, 14):
        mfcc_key = f"mfcc_{i}"
        row = f"  MFCC-{i:<21}"
        for _, feats in features_list:
            if mfcc_key in feats:
                mean = feats[mfcc_key]["mean"]
                std = feats[mfcc_key]["std"]
                row += f" | {mean:>10.4f} ± {std:<8.4f}"
            else:
                row += f" | {'N/A':>20}"
        print(row)

    print(f"{'=' * 80}\n")

    # 5. Сохранить JSON
    json_output = args.output_json or os.path.join(args.output_dir, "acoustic_features.json")
    save_json = [(name, feats) for name, feats in features_list]
    save_comparison_table(
        [{"name": n, "features": f} for n, f in save_json],
        json_output,
    )

    # 6. Построить графики
    # Для графиков нужны полные features (с мел-спектрограммами)
    plot_data = []
    if args.ref:
        ref_feats = extract_features_from_directory(args.ref, sr=args.sr, max_files=3)
        if ref_feats:
            plot_data.append(("Reference", compute_statistics_comparison(ref_feats, "Reference")))
    if args.gen_t2:
        t2_feats = extract_features_from_directory(args.gen_t2, sr=args.sr, max_files=3)
        if t2_feats:
            plot_data.append(("Tacotron 2", compute_statistics_comparison(t2_feats, "Tacotron 2")))
    if args.gen_vits:
        vits_feats = extract_features_from_directory(args.gen_vits, sr=args.sr, max_files=3)
        if vits_feats:
            plot_data.append(("VITS", compute_statistics_comparison(vits_feats, "VITS")))
    if args.gen and not args.gen_t2 and not args.gen_vits:
        gen_feats = extract_features_from_directory(args.gen, sr=args.sr, max_files=3)
        if gen_feats:
            plot_data.append((os.path.basename(args.gen), compute_statistics_comparison(gen_feats, os.path.basename(args.gen))))

    if plot_data:
        plot_comparison(plot_data, args.output_dir)

    print("[INFO] Анализ завершён!")


if __name__ == "__main__":
    main()
