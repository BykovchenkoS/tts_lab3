import re
import inflect
import unicodedata


# Проверка гласных
try:
    from unidecode import unidecode

    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False


_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9.,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")
# Паттерн для разделения текста на предложения
_sentence_split_re = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n"
    # Дополнительно: разделение по двоеточию и тире для длинных предложений
    r"|(?<=:)\s+(?=[A-Z])|(?<=\u2014)\s+(?=[A-Z])"
)


def normalize_numbers(text: str) -> str:
    """Заменить числа на их текстовое представление."""
    text = _comma_number_re.sub(lambda m: _inflect.number_to_words(m.group(1).replace(",", "")), text)
    text = _pounds_re.sub(lambda m: _inflect.number_to_words(m.group(1), only_standard=True) + " pounds", text)
    text = _dollars_re.sub(lambda m: _inflect.number_to_words(m.group(1)) + " dollars", text)
    text = _decimal_number_re.sub(lambda m: _inflect.number_to_words(m.group(1)), text)
    text = _ordinal_re.sub(lambda m: _inflect.number_to_words(m.group(0)), text)
    text = _number_re.sub(lambda m: _inflect.number_to_words(m.group(0)), text)
    return text


def unicode_to_ascii(text: str) -> str:
    """Конвертировать unicode в ascii."""
    if HAS_UNIDECODE:
        return unidecode(text)
    return text


def clean_text(text: str) -> str:
    """
    Полная очистка текста для TTS:
    1. Нормализация пробелов
    2. Удаление лишних символов
    3. Конвертация чисел в слова
    """
    # Нормализация unicode
    text = unicode_to_ascii(text)
    # Нормализация чисел
    text = normalize_numbers(text)
    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    """
    Разбить текст на предложения.
    Сохраняет знаки препинания (.!?) на конце каждого предложения.
    """
    # Добавляем специальный маркер для разделения
    sentences = []
    # Сначала разбиваем по стандартным разделителям
    parts = re.split(r"(?<=[.!?])\s+", text.strip())

    current = ""
    for part in parts:
        if not part:
            continue
        current = current + " " + part if current else part
        # Если часть заканчивается на знак препинания — это отдельное предложение
        if current.strip() and current.strip()[-1] in ".!?":
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences


def classify_sentence(sentence: str) -> str:
    """Определить тип предложения по интонации."""
    s = sentence.strip()
    if s.endswith("?"):
        return "question"
    elif s.endswith("!"):
        return "exclamation"
    elif ":" in s:
        return "colon"
    elif "\u2014" in s or "-" in s:
        return "dash"
    else:
        return "neutral"


def load_test_text(path: str) -> list[dict]:
    """
    Загрузить тестовый текст и разбить на предложения.
    Возвращает список словарей:
    [{"text": "...", "type": "question|neutral|..."}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)
    result = []
    for sent in sentences:
        cleaned = clean_text(sent)
        if cleaned:
            result.append({
                "text": cleaned,
                "type": classify_sentence(cleaned),
            })
    return result


def english_cleaners(text: str) -> str:
    """
    Базовый cleaner для английского текста (аналог из NVIDIA Tacotron2).
    """
    text = unicode_to_ascii(text)
    text = normalize_numbers(text)
    text = re.sub(r"[^ a-zA-Z'.,!?;:\-\u2014]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
