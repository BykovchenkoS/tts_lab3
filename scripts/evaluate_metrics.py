import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.audio_utils import load_audio, mel_spectrogram, resample_audio


def compute_pesq(ref_wav: np.ndarray, deg_wav: np.ndarray, sr: int = 16000) -> float:
    """
    PESQ (Perceptual Evaluation of Speech Quality).
    Оценивает perceptual quality: от -0.5 до 4.5, нормативная шкала 1.0–4.5.
    Требует sr = 16000.
    """
    try:
        from pesq import pesq as pesq_fn
        # PESQ требует 16000 Hz
        if sr != 16000:
            ref_16k = resample_audio(ref_wav, sr, 16000)
            deg_16k = resample_audio(deg_wav, sr, 16000)
        else:
            ref_16k = ref_wav
            deg_16k = deg_wav

        # Длины должны совпадать
        min_len = min(len(ref_16k), len(deg_16k))
        ref_16k = ref_16k[:min_len]
        deg_16k = deg_16k[:min_len]

        score = pesq_fn(16000, ref_16k, deg_16k, "wb")
        return float(score)
    except Exception as e:
        print(f"  [WARN] PESQ ошибка: {e}")
        return None


def compute_stoi(ref_wav: np.ndarray, deg_wav: np.ndarray, sr: int = 16000) -> float:
    """
    STOI (Short-Time Objective Intelligibility).
    Оценивает разборчивость: от 0.0 до 1.0, чем выше тем лучше.
    """
    try:
        from pystoi.stoi import stoi as stoi_fn
        if sr != 16000:
            ref_16k = resample_audio(ref_wav, sr, 16000)
            deg_16k = resample_audio(deg_wav, sr, 16000)
        else:
            ref_16k = ref_wav
            deg_16k = deg_wav

        min_len = min(len(ref_16k), len(deg_16k))
        ref_16k = ref_16k[:min_len]
        deg_16k = deg_16k[:min_len]

        score = stoi_fn(ref_16k, deg_16k, 16000, extended=False)
        return float(score)
    except Exception as e:
        print(f"  [WARN] STOI ошибка: {e}")
        return None


def compute_ssim_mel(ref_wav: np.ndarray, deg_wav: np.ndarray, sr: int = 22050) -> float:
    """
    SSIM мел-спектрограмм между эталонным и сгенерированным аудио.
    Используем sklearn structural_similarity.
    """
    try:
        from skimage.metrics import structural_similarity as ssim

        mel_ref = mel_spectrogram(ref_wav, sr=sr)
        mel_deg = mel_spectrogram(deg_wav, sr=sr)

        # Обрезаем до одинаковой длины по временной оси
        min_t = min(mel_ref.shape[1], mel_deg.shape[1])
        mel_ref = mel_ref[:, :min_t]
        mel_deg = mel_deg[:, :min_t]

        score = ssim(mel_ref, mel_deg, data_range=mel_ref.max() - mel_ref.min())
        return float(score)
    except ImportError:
        print("  [WARN] skimage не установлен, SSIM пропускается. pip install scikit-image")
        return None
    except Exception as e:
        print(f"  [WARN] SSIM ошибка: {e}")
        return None


def compute_mos_predicted(wav: np.ndarray, sr: int = 16000) -> dict:
    """
    Предсказанная MOS через SpeechMOS (необязательная метрика).
    """
    try:
        from speechmos import dnsmos

        if sr != 16000:
            wav_16k = resample_audio(wav, sr, 16000)
        else:
            wav_16k = wav

        mos = dnsmos.run(wav_16k, 16000)
        return {
            "mos_sig": float(mos.get("sig", 0)),
            "mos_bak": float(mos.get("bak", 0)),
            "mos_ovr": float(mos.get("ovr", 0)),
        }
    except Exception as e:
        print(f"  [WARN] MOS ошибка: {e}")
        return None


def evaluate_single_file(
    ref_wav_path: str,
    gen_wav_path: str,
    sr: int = 22050,
) -> dict:
    """Вычислить все метрики для одной пары файлов."""
    ref_wav = load_audio(ref_wav_path, sr=sr)
    gen_wav = load_audio(gen_wav_path, sr=sr)

    metrics = {
        "reference": os.path.basename(ref_wav_path),
        "generated": os.path.basename(gen_wav_path),
        "pesq": None,
        "stoi": None,
        "ssim_mel": None,
        "mos": None,
    }

    # PESQ
    pesq_score = compute_pesq(ref_wav, gen_wav, sr)
    metrics["pesq"] = pesq_score

    # STOI
    stoi_score = compute_stoi(ref_wav, gen_wav, sr)
    metrics["stoi"] = stoi_score

    # SSIM мел-спектрограмм
    ssim_score = compute_ssim_mel(ref_wav, gen_wav, sr)
    metrics["ssim_mel"] = ssim_score

    # MOS (только для сгенерированного)
    mos_score = compute_mos_predicted(gen_wav, sr)
    metrics["mos"] = mos_score

    return metrics


def evaluate_directory(
    gen_dir: str,
    ref_dir: str = None,
    results_file: str = None,
):
    """
    Оценить все файлы в директории.
    Если ref_dir не указан — вычисляем только MOS для сгенерированных файлов.
    """
    import glob

    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.wav")))
    if not gen_files:
        print(f"[ERROR] Нет .wav файлов в {gen_dir}")
        return []

    print(f"\n[INFO] Найдено {len(gen_files)} файлов в {gen_dir}")

    all_metrics = []

    for i, gen_path in enumerate(gen_files):
        filename = os.path.basename(gen_path)
        print(f"  [{i + 1}/{len(gen_files)}] {filename}")

        if ref_dir:
            # Ищем соответствующий референсный файл
            ref_path = os.path.join(ref_dir, filename)
            if not os.path.exists(ref_path):
                print(f"    [WARN] Референс не найден: {ref_path}, пропуск PESQ/STOI/SSIM")
                metrics = {"generated": filename, "pesq": None, "stoi": None, "ssim_mel": None}
                metrics["mos"] = compute_mos_predicted(load_audio(gen_path), sr=22050)
            else:
                metrics = evaluate_single_file(ref_path, gen_path, sr=22050)
        else:
            # Только MOS (без референса)
            gen_wav = load_audio(gen_path, sr=22050)
            metrics = {
                "generated": filename,
                "pesq": None,
                "stoi": None,
                "ssim_mel": None,
                "mos": compute_mos_predicted(gen_wav, sr=22050),
            }

        all_metrics.append(metrics)

        # Вывод метрик
        parts = []
        if metrics.get("pesq") is not None:
            parts.append(f"PESQ={metrics['pesq']:.3f}")
        if metrics.get("stoi") is not None:
            parts.append(f"STOI={metrics['stoi']:.3f}")
        if metrics.get("ssim_mel") is not None:
            parts.append(f"SSIM={metrics['ssim_mel']:.3f}")
        if metrics.get("mos"):
            parts.append(f"MOS={metrics['mos'].get('ovr', 0):.2f}")
        print(f"    {' | '.join(parts)}")

    # Сохраняем результаты
    if results_file:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        report = {
            "model": os.path.basename(gen_dir),
            "num_files": len(all_metrics),
            "metrics": compute_summary(all_metrics),
            "per_file": all_metrics,
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"\n[INFO] Результаты сохранены: {results_file}")

    # Вывод сводки
    print_summary(all_metrics, model_name=os.path.basename(gen_dir))

    return all_metrics


def compute_summary(metrics_list: list[dict]) -> dict:
    """Вычислить средние значения метрик."""
    summary = {"pesq": {}, "stoi": {}, "ssim_mel": {}, "mos": {}}

    pesq_vals = [m["pesq"] for m in metrics_list if m.get("pesq") is not None]
    stoi_vals = [m["stoi"] for m in metrics_list if m.get("stoi") is not None]
    ssim_vals = [m["ssim_mel"] for m in metrics_list if m.get("ssim_mel") is not None]

    if pesq_vals:
        summary["pesq"] = {"mean": float(np.mean(pesq_vals)), "std": float(np.std(pesq_vals)), "min": float(np.min(pesq_vals)), "max": float(np.max(pesq_vals))}
    if stoi_vals:
        summary["stoi"] = {"mean": float(np.mean(stoi_vals)), "std": float(np.std(stoi_vals)), "min": float(np.min(stoi_vals)), "max": float(np.max(stoi_vals))}
    if ssim_vals:
        summary["ssim_mel"] = {"mean": float(np.mean(ssim_vals)), "std": float(np.std(ssim_vals)), "min": float(np.min(ssim_vals)), "max": float(np.max(ssim_vals))}

    # MOS
    mos_vals = [m["mos"]["ovr"] for m in metrics_list if m.get("mos")]
    if mos_vals:
        summary["mos"] = {"mean": float(np.mean(mos_vals)), "std": float(np.std(mos_vals)), "min": float(np.min(mos_vals)), "max": float(np.max(mos_vals))}

    return summary


def print_summary(metrics_list: list[dict], model_name: str = ""):
    """Вывести сводную таблицу метрик."""
    summary = compute_summary(metrics_list)

    print(f"\n{'=' * 60}")
    print(f"СВОДКА МЕТРИК — {model_name}")
    print(f"{'=' * 60}")
    print(f"{'Метрика':<15} {'Среднее':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    for metric_name in ["pesq", "stoi", "ssim_mel", "mos"]:
        vals = summary.get(metric_name, {})
        if vals:
            print(f"{metric_name:<15} {vals['mean']:>8.4f} {vals['std']:>8.4f} {vals['min']:>8.4f} {vals['max']:>8.4f}")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Оценка качества синтезированной речи")

    parser.add_argument("--gen_dir", type=str, default=None, help="Директория с сгенерированными аудио")
    parser.add_argument("--ref_dir", type=str, default=None, help="Директория с референсными аудио")
    parser.add_argument("--gen_tacotron2", type=str, default=None, help="Директория с аудио Tacotron 2")
    parser.add_argument("--gen_vits", type=str, default=None, help="Директория с аудио VITS")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "data", "generated", "metrics_report.json"),
                        help="Путь к файлу с результатами")

    args = parser.parse_args()

    print("=" * 60)
    print("Оценка качества синтеза — PESQ, STOI, SSIM, MOS")
    print("=" * 60)

    results_all = {}

    # Оценка одной модели
    if args.gen_dir:
        metrics = evaluate_directory(
            gen_dir=args.gen_dir,
            ref_dir=args.ref_dir,
            results_file=args.output,
        )
        results_all[os.path.basename(args.gen_dir)] = metrics

    # Сравнительная оценка двух моделей
    elif args.gen_tacotron2 or args.gen_vits:
        output_dir = os.path.dirname(args.output)

        if args.gen_tacotron2:
            print("\n" + "=" * 60)
            print("Оценка Tacotron 2")
            print("=" * 60)
            t2_metrics = evaluate_directory(
                gen_dir=args.gen_tacotron2,
                ref_dir=args.ref_dir,
                results_file=os.path.join(output_dir, "metrics_tacotron2.json"),
            )
            results_all["tacotron2"] = t2_metrics

        if args.gen_vits:
            print("\n" + "=" * 60)
            print("Оценка VITS")
            print("=" * 60)
            vits_metrics = evaluate_directory(
                gen_dir=args.gen_vits,
                ref_dir=args.ref_dir,
                results_file=os.path.join(output_dir, "metrics_vits.json"),
            )
            results_all["vits"] = vits_metrics

        # Сравнительная таблица
        if len(results_all) == 2:
            print("\n" + "=" * 60)
            print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
            print("=" * 60)
            print(f"{'Метрика':<15} {'Tacotron 2':>15} {'VITS':>15}")
            print("-" * 60)
            for metric_name in ["pesq", "stoi", "ssim_mel", "mos"]:
                t2_summary = compute_summary(results_all["tacotron2"])
                vits_summary = compute_summary(results_all["vits"])
                t2_val = t2_summary.get(metric_name, {}).get("mean", "—")
                vits_val = vits_summary.get(metric_name, {}).get("mean", "—")
                if isinstance(t2_val, float):
                    t2_str = f"{t2_val:.4f}"
                else:
                    t2_str = str(t2_val)
                if isinstance(vits_val, float):
                    vits_str = f"{vits_val:.4f}"
                else:
                    vits_str = str(vits_val)
                print(f"{metric_name:<15} {t2_str:>15} {vits_str:>15}")
            print("=" * 60)

    else:
        print("[INFO] Использование:")
        print("  Оценить одну модель:  --gen_dir data/generated/tacotron2 --ref_dir data/raw/LJSpeech-1.1/wavs")
        print("  Сравнить две модели:  --gen_tacotron2 data/generated/tacotron2 --gen_vits data/generated/vits")
        print("  Без референса:        --gen_dir data/generated/vits  (только MOS)")


if __name__ == "__main__":
    main()
