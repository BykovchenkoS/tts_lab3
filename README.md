# Лабораторная работа №3 — Построение систем синтеза речи

## Модели
- **Tacotron 2** — классическая двухэтапная архитектура (текст → мел-спектрограмма + вокодер)
- **VITS** — современная энд-то-энд модель с GAN-дискриминатором

## Установка и запуск

### Шаг 1. Основные зависимости
```bash
pip install -r requirements.txt
pip install TTS --no-deps
```

### Шаг 2. Зависимости Coqui TTS
```bash
pip install cython g2p_en anyascii configargparse jamo jieba num2words phonemizer spacy tgt transformers unidic pesq pystoi
```

### Шаг 3. Зависимости Coqui TTS (продолжение)
```bash
pip install coqpit einops encodec flask gruut pysbd trainer umap-learn
pip install gruut
```

### Шаг 4. torchaudio (строго под версию torch)
```bash
pip install torchaudio==2.6.0
```

### Шаг 5. Фонемизаторы (TTS импортирует ВСЕ при старте)
```bash
pip install bangla bnnumerizer bnunicodenormalizer g2pkk hangul_romanize pypinyin
```

### Шаг 6. Скачать датасет
```bash
python scripts/prepare_data.py
```
Скачает LJSpeech (~2.5 GB) в `data/raw/LJSpeech-1.1` и создаст `metadata.csv` в формате 3 колонок.

### Шаг 7. Обучение Tacotron 2
```bash
# Быстрый тест (2 эпохи, 64 сэмпла):
python scripts/train_tacotron2.py --epochs 2 --batch_size 32 --small 64

# Полное обучение:
python scripts/train_tacotron2.py --epochs 1000 --batch_size 32
```

### Шаг 8. Обучение VITS
```bash
# Быстрый тест (2 эпохи, 64 сэмпла):
python scripts/train_vits.py --epochs 2 --batch_size 32 --small 64

# Полное обучение:
python scripts/train_vits.py --epochs 1000 --batch_size 32
```

### Шаг 9. Синтез речи
```bash
python scripts/synthesize.py
```
Генерирует WAV-файлы в `output/synthesis/tacotron2/` и `output/synthesis/vits/`.

### Шаг 10. Оценка метрик
```bash
python scripts/evaluate_metrics.py
```
Считает PESQ, STOI, mel-SSIM. Результаты в `output/evaluation/`.

### Шаг 11. Акустический анализ
```bash
python scripts/acoustic_analysis.py
```
Строит графики MFCC, F0, спектральных признаков. Результаты в `output/analysis/`.

### Шаг 12. TensorBoard
```bash
tensorboard --logdir run
```
Открыть в браузере `http://localhost:6006`. Показывает loss, learning rate, мел-спектрограммы.
