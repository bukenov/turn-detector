# Turn Detector

Детектор конца реплики (end-of-turn detection) для голосовых AI-агентов.

## Бизнес-задача

Голосовые AI-боты должны понимать, когда пользователь закончил свою мысль,
чтобы отвечать без задержки и не перебивать. Модель предсказывает по тексту
реплики: закончил пользователь или ещё договорит.

## Результаты

| Модель | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Baseline (правило по пунктуации) | 0.77 | 0.66 | 0.91 | — |
| Logistic Regression (tuned threshold) | 0.79 | 0.72 | 0.88 | 0.97 |
| LightGBM (default) | 0.82 | 0.77 | 0.88 | 0.98 |
| **LightGBM (Optuna tuned)** | **0.83** | 0.78 | 0.88 | 0.98 |

Финальная модель деградирует с F1=0.83 до 0.77 на out-of-domain данных
(DailyDialog). Главный drift — `last_pos` (PSI=1.06), но модель устойчива
благодаря `ends_with_punct` как доминирующему признаку.

## Стек

Python 3.13, pandas, scikit-learn, LightGBM, Optuna, spaCy, FastAPI, Docker.

## Структура

```
turn-detector/
├── notebooks/
│   ├── 01_eda.ipynb           # Разведочный анализ
│   ├── 02_modeling.ipynb      # 4 модели, Optuna
│   └── 03_drift.ipynb         # PSI и деградация на DailyDialog
├── src/
│   ├── features.py            # extract_features() + normalize_input()
│   └── app.py                 # FastAPI endpoints
├── models/turn_detector.pkl   # Финальная модель + артефакты
├── data/sample.csv            # Пример данных (500 строк)
├── reports/
│   ├── midterm_report.md      # Промежуточный отчёт
│   ├── final_report.md        # Финальный отчёт
│   ├── Turn-Detector.pdf      # Презентация Demo Day
│   └── figures/               # Графики для отчёта
├── Dockerfile
└── requirements.txt
```

## Запуск через Docker

```bash
docker build -t turn-detector:latest .
docker run --rm -p 8000:8000 turn-detector:latest
```

Swagger UI: http://localhost:8000/docs

## Запуск локально (без Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn src.app:app --port 8000
```

## API

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I would like to book a hotel"}'
```

Response:
```json
{
  "is_end_of_turn": true,
  "confidence": 0.94,
  "threshold": 0.87,
  "features": {"length_words": 7, "last_pos": "PROPN", ...}
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

## Воспроизведение модели

```bash
jupyter notebook
# Запусти 01_eda.ipynb → создаст data/processed.parquet
# Запусти 02_modeling.ipynb → создаст models/turn_detector.pkl
# Запусти 03_drift.ipynb → drift-анализ
```

## Известные ограничения

- Модель сильно опирается на `ends_with_punct` (importance в 10× выше
  следующего признака). На фразах без пунктуации могла бы недоопределять
  конец, но это компенсируется input normalization в API (`src/features.py:normalize_input`),
  которая добавляет терминальную пунктуацию по эвристике.
- Train domain — booking-диалоги (MultiWOZ). На разговорной речи
  деградация ~6% F1 (см. drift-анализ).

## Дальнейшие улучшения

- Sentence embeddings вместо хэндкрафтнутых признаков
- Расширение train-данных конверсационными корпусами (Switchboard, DailyDialog)
- Multi-stage Docker build для уменьшения размера образа
- A/B тест в реальной голосовой системе

## Author

Курсовой проект: AI Engineer — Machine Learning, март 2026.
