# punctuation_model.py

import torch

# Загрузка модели
model, example_texts, languages, punct, apply_te = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_te'
)

def process_text(text, language='en'):
    """
    Обрабатывает текст с помощью модели расстановки знаков препинания.
    """
    return apply_te(text, lan=language)