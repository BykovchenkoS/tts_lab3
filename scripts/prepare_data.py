import os
import sys
import argparse
import tarfile
import urllib.request
import random

# Добавляем корень проекта в путь
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJSPEECH_MD5 = "be1a30453f28a39b027353c244237e91"


def download_ljspeech(output_dir: str):
    """Скачать и распаковать LJSpeech датасет."""
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, "LJSpeech-1.1.tar.bz2")

    if os.path.exists(os.path.join(output_dir, "LJSpeech-1.1")):
        print(f"[INFO] LJSpeech уже существует в {output_dir}/LJSpeech-1.1")
        return

    if not os.path.exists(archive_path):
        print(f"[INFO] Скачивание LJSpeech ({LJSPEECH_URL})...")
        # Используем tqdm-like прогресс-бар через urllib
        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r[INFO] Скачано: {mb_down:.1f} / {mb_total:.1f} MB ({pct:.0f}%)", end="")

        urllib.request.urlretrieve(LJSPEECH_URL, archive_path, reporthook=report)
        print()  # Перевод строки после прогресс-бара

    print(f"[INFO] Распаковка {archive_path}...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(output_dir)

    print(f"[INFO] Готово! Датасет в {output_dir}/LJSpeech-1.1")


def parse_metadata(metadata_path: str) -> list[dict]:
    """
    Парсинг метаданных LJSpeech (metadata.csv).
    Формат: file_id|normalized_text|original_text
    Возвращает список: [{"file_id": "...", "text": "..."}, ...]
    """
    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                file_id = parts[0]
                text = parts[2] if len(parts) > 2 else parts[1]
                items.append({"file_id": file_id, "text": text})
    return items


def split_dataset(
    items: list[dict],
    val_ratio: float = 0.05,
    seed: int = 1234,
) -> tuple[list[dict], list[dict]]:
    """Разбить датасет на train/val."""
    random.seed(seed)
    indices = list(range(len(items)))
    random.shuffle(indices)

    val_size = max(1, int(len(items) * val_ratio))
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    train_items = [items[i] for i in sorted(train_indices)]
    val_items = [items[i] for i in sorted(val_indices)]

    return train_items, val_items


def save_splits(train_items: list[dict], val_items: list[dict], output_dir: str):
    """Сохранить train/val сплиты в формате CSV для Coqui TTS."""
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем в формате: file_id|text
    for split_name, items in [("train", train_items), ("val", val_items)]:
        path = os.path.join(output_dir, f"{split_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(f"{item['file_id']}|{item['text']}\n")
        print(f"[INFO] {split_name}: {len(items)} записей -> {path}")

    # Также сохраняем полные метаданные
    meta_path = os.path.join(output_dir, "metadata.csv")
    all_items = train_items + val_items
    with open(meta_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(f"{item['file_id']}|{item['text']}\n")
    print(f"[INFO] Полные метаданные: {len(all_items)} записей -> {meta_path}")


def print_dataset_stats(items: list[dict], data_dir: str, wavs_dir: str):
    """Вывести статистику по датасету."""
    print("\n" + "=" * 60)
    print("Статистика датасета LJSpeech")
    print("=" * 60)
    print(f"  Количество записей: {len(items)}")

    # Подсчёт длительности
    total_duration = 0.0
    missing = 0
    for item in items:
        wav_path = os.path.join(wavs_dir, f"{item['file_id']}.wav")
        if os.path.exists(wav_path):
            try:
                import librosa
                wav, sr = librosa.load(wav_path, sr=None)
                total_duration += len(wav) / sr
            except Exception:
                missing += 1
        else:
            missing += 1

    hours = total_duration / 3600
    print(f"  Общая длительность: {hours:.1f} часов ({total_duration:.0f} секунд)")
    if missing > 0:
        print(f"  Пропущенных файлов: {missing}")

    # Средняя длина текста
    text_lengths = [len(item["text"].split()) for item in items]
    print(f"  Средняя длина текста: {sum(text_lengths) / len(text_lengths):.1f} слов")
    print(f"  Мин / Макс длина текста: {min(text_lengths)} / {max(text_lengths)} слов")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Подготовка датасета LJSpeech")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Директория для сохранения датасета (default: data/raw)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Доля валидационной выборки (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234)",
    )
    args = parser.parse_args()

    # 1. Скачать
    download_ljspeech(args.output_dir)

    # 2. Парсинг
    data_dir = os.path.join(args.output_dir, "LJSpeech-1.1")
    metadata_path = os.path.join(data_dir, "metadata.csv")

    if not os.path.exists(metadata_path):
        print(f"[ERROR] Не найден {metadata_path}")
        sys.exit(1)

    items = parse_metadata(metadata_path)
    print(f"[INFO] Загружено {len(items)} записей из metadata.csv")

    # 3. Train/Val split
    train_items, val_items = split_dataset(items, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[INFO] Train: {len(train_items)}, Val: {len(val_items)}")

    # 4. Сохранить сплиты
    save_splits(train_items, val_items, data_dir)

    # 5. Статистика
    wavs_dir = os.path.join(data_dir, "wavs")
    print_dataset_stats(train_items, data_dir, wavs_dir)

    print("[INFO] Подготовка датасета завершена!")


if __name__ == "__main__":
    main()
