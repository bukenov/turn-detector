# Turn Detector

Детектор конца реплики (end-of-turn detection) для голосовых AI-агентов.

## Бизнес-задача

Голосовые AI-боты должны понимать, когда пользователь закончил свою мысль,
чтобы отвечать без задержки и не перебивать. Эта модель предсказывает по
тексту реплики: закончил пользователь или ещё договорит.

## Датасет

[MultiWOZ 2.2](https://huggingface.co/datasets/pfb30/multi_woz_v22) — ~8400
диалогов про бронирование отелей, ресторанов, такси.

## Стек

Python 3.13, pandas, scikit-learn, lightgbm, spaCy, HuggingFace datasets.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook notebooks/01_eda.ipynb
```

## Структура

- `notebooks/` — Jupyter-блокноты с EDA, моделями, drift-анализом
- `src/` — переиспользуемый код (заполнится по мере проекта)
- `data/sample.csv` — пример данных (500 строк)
- `reports/` — промежуточный отчёт и финальные графики

## Статус

В разработке. Курсовой проект AI Engineer ML, март 2026.
