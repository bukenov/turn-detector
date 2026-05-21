"""
Извлечение признаков для модели turn detector.
Используется и при обучении (в notebooks), и при инференсе (в app.py).
"""
from typing import Optional
import pandas as pd
import spacy


QUESTION_WORDS = {"what", "where", "when", "who", "why", "how", "which", "whose"}

NUMERIC_FEATURES = [
    "length_words",
    "length_chars",
    "ends_with_punct",
    "ends_with_prep",
    "has_question_word",
]
CATEGORICAL_FEATURES = ["last_pos"]


_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    """Ленивая загрузка spaCy — один раз на процесс."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(
            "en_core_web_sm",
            disable=["ner", "parser", "lemmatizer"],
        )
    return _nlp


def extract_features(text: str) -> dict:
    """
    Превращает текст префикса в словарь признаков.
    
    Возвращает dict со столбцами, ожидаемыми моделью.
    """
    text = text.strip()
    if not text:
        return {
            "length_words": 0,
            "length_chars": 0,
            "ends_with_punct": 0,
            "ends_with_prep": 0,
            "has_question_word": 0,
            "last_pos": "X",
        }
    
    words = text.split()
    last_word = words[-1].lower()
    
    nlp = _get_nlp()
    doc = nlp(last_word)
    last_pos = doc[0].pos_ if len(doc) > 0 else "X"
    
    return {
        "length_words": len(words),
        "length_chars": len(text),
        "ends_with_punct": int(text.rstrip()[-1] in ".?!"),
        "ends_with_prep": int(last_pos == "ADP"),
        "has_question_word": int(
            any(w in text.lower().split() for w in QUESTION_WORDS)
        ),
        "last_pos": last_pos,
    }


def features_to_dataframe(features: dict, ohe, feature_columns: list) -> pd.DataFrame:
    """
    Превращает dict признаков в DataFrame с правильным OHE,
    в том же порядке столбцов что и при обучении.
    """
    df = pd.DataFrame([features])
    
    # OHE для категориальных
    ohe_array = ohe.transform(df[CATEGORICAL_FEATURES])
    ohe_columns = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_df = pd.DataFrame(ohe_array, columns=ohe_columns, index=df.index)
    
    # Финальная сборка в правильном порядке
    result = pd.concat([df[NUMERIC_FEATURES], ohe_df], axis=1)
    
    # Гарантируем порядок столбцов из обучения
    result = result.reindex(columns=feature_columns, fill_value=0)
    
    return result
