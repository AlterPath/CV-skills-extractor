import random

import torch
from torch import nn

from competency_match.tagger import (B, I, IGNORE, LABELS, O, SkillTagger, TaggerRunner, Vocab, build_corpus,
                                     encode_batch, entity_spans, make_sentence, train)


def test_entity_spans_from_bio():
    assert entity_spans([O, B, I, O, B, O, I, I]) == {(1, 3), (4, 5), (6, 8)}


def test_sentence_labels_follow_inserted_skills():
    tokens, labels = make_sentence("Стек: {s}, SQL.", ["Apache Spark"], random.Random(0))
    assert [t.lower() for t in tokens] == ["стек", ":", "apache", "spark", ",", "sql"]
    assert labels == [O, O, B, I, O, O]


def test_padding_is_ignored_and_o_counts():
    """Ошибка v1: метка O совпадала с паддингом, и при ignore_index=0 фон не учитывался, а лосс бывал NaN."""
    batch = [(["Английский", "язык"], [O, O]), (["Python"], [B])]
    vocab = Vocab.build(batch)
    *_, labels = encode_batch(batch, vocab)
    assert labels.tolist() == [[O, O], [B, IGNORE]]
    model = SkillTagger(len(vocab.words), len(vocab.chars))
    words, chars, feats, lengths, labels = encode_batch(batch[:1], vocab)    # в батче одни O
    loss = nn.CrossEntropyLoss(ignore_index=IGNORE)(model(words, chars, feats, lengths).view(-1, len(LABELS)),
                                                   labels.view(-1))
    assert torch.isfinite(loss) and loss.item() > 0


def test_tiny_training_learns_skills():
    rng = random.Random(1)
    pool = ["Python", "Docker", "PostgreSQL", "машинное обучение", "Kafka"]
    data = build_corpus(pool, 300, rng)
    vocab = Vocab.build(data)
    model = train(data, data[:60], vocab, epochs=4, log=lambda *_: None)
    spans = TaggerRunner(model, vocab).spans("Опыт работы с Docker и Kafka.")
    assert {fragment for _, _, fragment in spans} == {"Docker", "Kafka"}
