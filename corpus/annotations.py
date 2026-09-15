import re

from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import filter_spans
from terms import GROUP_TERMS, PROPERTY_TERMS, USE_TERMS

if not Doc.has_extension("filtered_tokens"):
    Doc.set_extension("filtered_tokens", default=None)


@Language.component("snhu_tokenizer")
def snhu_tokenizer(doc: Doc) -> Doc:
    doc._.filtered_tokens = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return doc


def find_spans(text: str, terms: list[str], label: str):
    spans = []
    for term in terms:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        matches = pattern.finditer(text)
        for match in matches:
            spans.append((match.start(), match.end(), label))
    return spans


def annotate_doc(
    nlp: Language,
    title: str,
    sentence_text: str,
):

    text = sentence_text.strip()
    doc = nlp.make_doc(text)

    if not text:
        return doc

    annotations = []

    annotations.extend(
        find_spans(
            text,
            [title],
            "MINERAL_NAME",
        )
    )

    annotations.extend(
        find_spans(
            text,
            GROUP_TERMS,
            "MINERAL_GROUP",
        )
    )

    annotations.extend(
        find_spans(
            text,
            PROPERTY_TERMS,
            "MINERAL_PROPERTY",
        )
    )

    annotations.extend(
        find_spans(
            text,
            USE_TERMS,
            "MINERAL_USE",
        )
    )

    spans = []

    for start, end, label in annotations:
        span = doc.char_span(
            start,
            end,
            label=label,
            alignment_mode="expand",
        )

        if span is not None:
            spans.append(span)

    doc.ents = filter_spans(spans)

    return doc


def normalize_text(text: str):
    return text.casefold().strip()
