import numpy as np
import torch
import librosa
import soundfile as sf
import os


def load_audio(path: str, sr: int = 22050) -> np.ndarray:
    """Загрузить аудиофайл и ресемплировать до sr."""
    wav, _ = librosa.load(path, sr=sr)
    return wav


def save_audio(wav: np.ndarray, path: str, sr: int = 22050):
    """Сохранить аудиофайл (wav)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sr)


def mel_spectrogram(
    wav: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Вычислить мел-кепстральную спектрограмму из аудиосигнала.
    Возвращает массив shape (n_mels, T).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    # Конвертация в лог-масштаб (log mel spectrogram)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel


def mel_spectrogram_torch(
    wav: torch.Tensor,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> torch.Tensor:
    """
    Вычислить мел-спектрограмму с использованием torchaudio.
    Возвращает тензор shape (n_mels, T).
    """
    import torchaudio.transforms as T

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    return mel_transform(wav)


def get_audio_duration(path: str) -> float:
    """Получить длительность аудиофайла в секундах."""
    wav, sr = librosa.load(path, sr=None)
    return len(wav) / sr


def compute_spectral_features(wav: np.ndarray, sr: int = 22050):
    """
    Вычислить набор спектральных признаков для акустического анализа:
    - spectral centroid (центроид)
    - spectral bandwidth (ширина полосы)
    - spectral rolloff (rolloff)
    - spectral contrast (спектральный контраст)
    - spectral flatness (плоскостность)
    - zero crossing rate (частота переходов через ноль)
    - RMS energy
    - MFCC (13 коэффициентов)
    - pitch (F0) через pyin
    """
    features = {}

    # Спектральные признаки
    features["spectral_centroid"] = librosa.feature.spectral_centroid(y=wav, sr=sr)[0]
    features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(y=wav, sr=sr)[0]
    features["spectral_rolloff"] = librosa.feature.spectral_rolloff(y=wav, sr=sr)[0]
    features["spectral_contrast"] = librosa.feature.spectral_contrast(y=wav, sr=sr)
    features["spectral_flatness"] = librosa.feature.spectral_flatness(y=wav)[0]
    features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(wav)[0]
    features["rms"] = librosa.feature.rms(y=wav)[0]

    # MFCC (13 коэффициентов)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i + 1}"] = mfcc[i]

    # Основная частота (F0) через pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    features["f0"] = f0
    features["voiced_flag"] = voiced_flag
    features["voiced_probs"] = voiced_probs

    return features


def resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Ресемплировать аудио."""
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
