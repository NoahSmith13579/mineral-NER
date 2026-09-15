import json
from pathlib import Path

import spacy
from config.settings import corpus_directory
from spacy.tokens import DocBin


def load_corpus():
    nlp = spacy.load("en_core_web_lg")

    print("Loading Corpus...")
    corpus = {
        "train": list(
            DocBin().from_disk(corpus_directory / "train.spacy").get_docs(nlp.vocab)
        ),
        "validation": list(
            DocBin()
            .from_disk(corpus_directory / "validation.spacy")
            .get_docs(nlp.vocab)
        ),
        "test": list(
            DocBin().from_disk(corpus_directory / "test.spacy").get_docs(nlp.vocab)
        ),
    }
    print("Corpus loaded!")
    return corpus


def load_sentence_corpus(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records
