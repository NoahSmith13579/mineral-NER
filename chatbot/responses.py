from minerals import minerals
from retrieval.sentence_retrieval import retrieve_relevant_sentence


def model_response(question: str, nlp, sentence_records: list[dict]) -> str:
    question_doc = nlp(question)
    entities = {
        "MINERAL_NAME": [],
        "MINERAL_GROUP": [],
        "MINERAL_PROPERTY": [],
        "MINERAL_USE": [],
    }
    for entity in question_doc.ents:
        if entity.label_ in entities:
            entities[entity.label_].append(entity.text)

    minerals_found = entities["MINERAL_NAME"].copy()
    if not minerals_found:
        for name in minerals:
            if name.casefold() in question.casefold():
                minerals_found.append(name)
                break

    if not minerals_found:
        return "I could not identify a mineral in the question."

    sentence = retrieve_relevant_sentence(
        question=question,
        nlp=nlp,
        mineral=minerals_found[0],
        entities=entities,
        sentence_records=sentence_records,
    )
    return sentence or "I could not find a relevant fact."


def rule_response(question: str) -> str | None:
    text = question.casefold().strip()
    if text == "what is your subject?":
        return "Mineralogy"
    if text == "what are the primary objects you know about?":
        return "I know about minerals and their properties"
    if "how many records" in text and "train" in text:
        return "I trained with 16745 records."
    if "last updated" in text:
        return "I was updated at: September 3, 2026."
    return None


def answer_question(question: str, nlp, sentence_records: list[dict]) -> dict:
    rule_answer = rule_response(question)
    if rule_answer is not None:
        return {"answer": rule_answer, "response_type": "rule-based"}
    return {
        "answer": model_response(question, nlp, sentence_records),
        "response_type": "model-based",
    }
