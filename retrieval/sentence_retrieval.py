from corpus.annotations import normalize_text
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language


def retrieve_relevant_sentence(
    question: str,
    nlp: Language,
    mineral: str,
    entities: dict[str, list[str]],
    sentence_records: list[dict],
) -> str | None:

    question_text = normalize_text(question)
    mineral_text = normalize_text(mineral)

    possible_sentences: list[dict] = [
        record
        for record in sentence_records
        if (
            str(record.get("title", "")) == mineral_text
            and str(record.get("sentence", "")).strip()
        )
    ]

    if not possible_sentences:
        return None

    retrieval_terms = [question_text]

    for label in [
        "MINERAL_GROUP",
        "MINERAL_PROPERTY",
        "MINERAL_USE",
    ]:
        for entity_text in entities[label]:
            retrieval_terms.append(normalize_text(entity_text))

    retrieval_query = " ".join(retrieval_terms)

    def snhu_tfidf_tokenizer(document: str):
        doc = nlp(document)

        return [
            token.lemma_.casefold()
            for token in doc
            if (not token.is_stop and not token.is_punct and not token.is_space)
        ]

    possible_texts = [str(record["sentence"]) for record in possible_sentences]

    vectorizer = TfidfVectorizer(
        tokenizer=snhu_tfidf_tokenizer,
        sublinear_tf=True,
        norm="l2",
    )
    results = vectorizer.fit_transform(possible_texts)
    question_vector = vectorizer.transform([retrieval_query])

    cos_sim: ndarray = cosine_similarity(question_vector, results)[0]

    best_index = cos_sim.argmax()
    best_score = cos_sim[best_index]

    if best_score <= 0.2:
        return None
    return str(possible_sentences[best_index]["sentence"])
