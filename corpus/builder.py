import json

import spacy
from config.settings import (
    corpus_directory,
    data_directory,
    pipeline_directory,
)
from datasets import Dataset, load_dataset
from minerals import minerals
from spacy.tokens import DocBin

from corpus.annotations import annotate_doc, normalize_text


def create_corpus() -> None:

    nlp = spacy.load(pipeline_directory / "minerals_tokenized.pipeline")

    print("Loading files from Wikipedia directory...")
    dataset = load_dataset(
        "parquet",
        data_dir=str(data_directory),
        data_files="train*.parquet",
        split="all",
    )
    print("Loaded!")
    mineral_titles = {str(item).strip().casefold() for item in minerals}
    print("Filtering dataset for only mineral articles...")
    dataset_mineral: Dataset = dataset.filter(
        lambda row: str(row["title"]).strip().casefold() in mineral_titles
    )
    print("Filtering complete!")

    sentence_path = data_directory / "mineral_sentence_corpus"
    print("Creating Sentence Corpus...")
    with sentence_path.open("w", encoding="utf-8") as file:
        for article_id, article in enumerate(dataset_mineral):
            title = normalize_text(str(article["title"]))  # type: ignore
            document = nlp(normalize_text(str(article["maintext"])))  # type: ignore
            for sentence_id, sentence in enumerate(document.sents):
                file.write(
                    json.dumps(
                        {
                            "article_id": article_id,
                            "title": title,
                            "sentence_id": sentence_id,
                            "sentence": normalize_text(sentence.text),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print("Sentence Corpus Saved!")

    training_split = dataset_mineral.train_test_split(test_size=0.30, seed=1)
    validation_split = training_split["test"].train_test_split(test_size=0.5, seed=1)
    corpus = {
        "train": training_split["train"],
        "validation": validation_split["train"],
        "test": validation_split["test"],
    }

    print("Saving splits to disk...")
    for key, split in corpus.items():
        doc_bin = DocBin()
        for document in split:
            title = normalize_text(str(document["title"]))  # type: ignore
            doc = nlp(str(document["maintext"]))  # type: ignore
            for sentence in doc.sents:
                doc_bin.add(annotate_doc(nlp, title, sentence.text))
        doc_bin.to_disk(corpus_directory / f"{key}.spacy")
    print("Splits Saved!")
