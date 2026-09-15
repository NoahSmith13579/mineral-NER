import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example


def evaluate_model(model: Language, docs: list[Doc]):
    return model.evaluate([Example(model(doc.text), doc) for doc in docs])


def test_model(test_split: list[Doc], best_model_path):
    model = spacy.load(best_model_path)
    scores = evaluate_model(model, test_split)
    print("\nOverall test performance")
    print(f"Precision: {scores['ents_p']:.3f}")
    print(f"Recall:    {scores['ents_r']:.3f}")
    print(f"F-score:   {scores['ents_f']:.3f}")
    print("\nPerformance by entity type")
    for label, metrics in scores["ents_per_type"].items():
        print(
            f"{label}: precision={metrics['p']:.3f}, "
            f"recall={metrics['r']:.3f}, f-score={metrics['f']:.3f}"
        )
