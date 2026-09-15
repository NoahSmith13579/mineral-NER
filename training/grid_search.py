import shutil
from itertools import product
from pathlib import Path

import spacy
from config.settings import (
    backend_directory,
    chat_model_path,
    config_path,
    grid_output_directory,
)
from spacy.cli.train import train
from spacy.tokens import DocBin

from training.evaluation import evaluate_model, test_model


def experiment(dropout: float, learning_rate: float, max_steps: int):
    name = f"dropout-{dropout}_rate-{learning_rate}_steps-{max_steps}"
    output_path = grid_output_directory / name
    print(f"\nStarting experiment: {name}")
    train(
        config_path=config_path,
        output_path=output_path,
        overrides={
            "training.dropout": dropout,
            "training.optimizer.learn_rate": learning_rate,
            "training.max_steps": max_steps,
        },
    )
    model = spacy.load(output_path / "model-best")
    validation = list(
        DocBin()
        .from_disk(backend_directory / "corpus" / "validation.spacy")
        .get_docs(model.vocab)
    )
    scores = evaluate_model(model, validation)
    return {
        "experiment": name,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "precision": scores["ents_p"],
        "recall": scores["ents_r"],
        "f_score": scores["ents_f"],
        "per_entity": scores["ents_per_type"],
        "model_path": str(output_path / "model-best"),
    }


def update_chat_model(path: str) -> None:
    destination = chat_model_path / "model-best"
    if destination.exists():
        shutil.rmtree(destination)
    chat_model_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(path), destination)


def grid_search() -> None:
    grid = {
        "dropout": [0.05, 0.10, 0.15],
        "learning_rate": [0.0005, 0.001],
        "max_steps": [400, 600, 800],
    }
    results = [
        experiment(dropout, rate, steps)
        for dropout, rate, steps in product(
            grid["dropout"],
            grid["learning_rate"],
            grid["max_steps"],
        )
    ]
    results.sort(key=lambda result: result["f_score"], reverse=True)
    for result in results:
        print(
            f"{result['experiment']}:\n"
            f" f1: {result['f_score']:.4f}\n"
            f" precision: {result['precision']:.4f}\n"
            f" Recall: {result['recall']:.4f}"
        )
    model = spacy.load(results[0]["model_path"])
    test_split = list(
        DocBin()
        .from_disk(backend_directory / "corpus" / "test.spacy")
        .get_docs(model.vocab)
    )
    test_model(test_split, results[0]["model_path"])
    update_chat_model(results[0]["model_path"])
