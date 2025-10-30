import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
import os
import json
import re


# Подготовка данных

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # batch - список кортежей (x_tensor, y_tensor) разной длины
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_padded, ys_padded


# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

def build_vocab(texts):
    word_to_ix = defaultdict(lambda: len(word_to_ix))
    word_to_ix["<PAD>"] = 0
    for text in texts:
        for word in text.lower().split():
            _ = word_to_ix[word]
    return word_to_ix

def annotate(text, competency_list):
    words = text.lower().split()
    labels = ["O"] * len(words)
    text_str = " ".join(words)
    for skill in competency_list:
        skill_words = skill.lower().split()
        # Создаем паттерн для поиска skill в тексте как отдельной последовательности слов
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in skill_words) + r"\b"
        match = re.search(pattern, text_str)
        if match:
            # Найдем позицию начала в терминах токенов
            start_char = match.start()
            start_idx = len(text_str[:start_char].split())
            labels[start_idx] = "B-COMP"
            for j in range(1, len(skill_words)):
                if start_idx + j < len(labels):
                    labels[start_idx + j] = "I-COMP"
    return words, labels


# Dataset


class ResumeSkillDataset(Dataset):
    def __init__(self, texts, competencies, word_to_ix, label_to_ix):
        self.data = []
        for text in texts:
            words, labels = annotate(text, competencies)
            word_ids = [word_to_ix.get(w, 0) for w in words]
            label_ids = [label_to_ix.get(l, 0) for l in labels]
            # Проверка на совпадение длины и непустоту
            if len(word_ids) == len(label_ids) and len(word_ids) > 0:
                self.data.append((word_ids, label_ids))
            else:
                print(f"Warning: skipping text due to length mismatch or empty tokens: {text}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

# Модель


class SkillTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SkillTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        out = self.fc(lstm_out)
        return out

# Обучение

def train_model(model, dataloader, label_pad_id=0, epochs=10):
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Уменьшили LR

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        label_freq = defaultdict(int)

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)  # [batch, seq_len, classes]
            output = output.view(-1, output.shape[-1])
            y_batch = y_batch.view(-1)

            # Вычисляем loss
            loss = criterion(output, y_batch)
            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping this batch")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Подсчёт распределения меток в батче
            labels_flat = y_batch.tolist()
            for l in labels_flat:
                label_freq[l] += 1

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, label distribution: {dict(label_freq)}")

# Сравнение резюме с компетенциями

def compare_resume_with_competencies(resume_text, competency_objects):
    found = []
    missing = []
    resume_text_lower = resume_text.lower()

    for comp in competency_objects:
        skill = comp["name"].lower()
        if skill in resume_text_lower:
            found.append({
                "name": comp["name"],
                "level": comp["required_level_name"],
                "status": "найдено"
            })
        else:
            missing.append({
                "name": comp["name"],
                "level": comp["required_level_name"],
                "status": "нужно доучить"
            })

    return found, missing


def load_competencies_from_alliance_matrix(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    competencies = []
    for role_code, role_data in data.get("roles", {}).items():
        competencies.extend(role_data.get("competencies", []))
    return competencies


if __name__ == "__main__":
    texts = [
        "Занимался разработкой моделей машинного обучения и использовал Docker",
        "Изучал определения, историю развития и главные тренды ИИ, работал с Git и Bash"
    ]

    # Загрузка компетенций из JSON
    alliance_path = "alliance_matrix.json"
    if os.path.exists(alliance_path):
        competency_objects = load_competencies_from_alliance_matrix(alliance_path)
        competencies = [c["name"] for c in competency_objects]
    else:
        competency_objects = [
            {"name": "модели машинного обучения", "required_level_name": "Продвинутый"},
            {"name": "Docker", "required_level_name": "Базовый"},
            {"name": "искусственный интеллект", "required_level_name": "Базовый"},
            {"name": "Git", "required_level_name": "Базовый"},
            {"name": "Bash", "required_level_name": "Продвинутый"}
        ]
        competencies = [c["name"] for c in competency_objects]

    label_to_ix = {"O": 0, "B-COMP": 1, "I-COMP": 2}
    word_to_ix = build_vocab(texts)

    dataset = ResumeSkillDataset(texts, competencies, word_to_ix, label_to_ix)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = SkillTagger(vocab_size=len(word_to_ix), embedding_dim=64, hidden_dim=128, output_dim=len(label_to_ix))
    train_model(model, dataloader)

    resume_path = "sample_resume.txt"
    if os.path.exists(resume_path):
        with open(resume_path, "r", encoding="utf-8") as f:
            test_resume = f.read()
    else:
        test_resume = "Работал с Docker и Git, занимался ML"

    found, missing = compare_resume_with_competencies(test_resume, competency_objects)
    print("\nНайдено в резюме:")
    for f in found:
        print(f" - {f['name']} (уровень: {f['level']})")

    print("\nНужно доучить:")
    for m in missing:
        print(f" - {m['name']} (требуемый уровень: {m['level']})")
