import os
import threading
from contextlib import asynccontextmanager

import requests
import spacy
import uvicorn
from chatbot.responses import answer_question
from config.settings import (
    backend_directory,
    best_model_path,
    frontend_url,
    sentence_corpus_path,
)
from corpus.corpus_io import load_sentence_corpus
from fastapi import FastAPI

nlp = None
sentence_records = []

stop_polling = threading.Event()


def poll():

    while not stop_polling.is_set():
        try:
            if nlp is None:
                raise RuntimeError("The chatbot model has not been loaded.")

            resp = requests.get(f"{frontend_url}/api/v1/getQuestion", timeout=None)

            resp.raise_for_status()

            if resp.text:
                res = answer_question(resp.text, nlp, sentence_records)

                reply = requests.post(
                    f"{frontend_url}/api/v1/replytoQuestion",
                    data=res["answer"],
                    headers={"Content-Type": "text/plain"},
                    timeout=10,
                )

                reply.raise_for_status()

        except requests.RequestException as error:
            print(f"Backend error: {error}")
            stop_polling.wait(2)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global nlp, sentence_records

    nlp = spacy.load(best_model_path)
    sentence_records = load_sentence_corpus(sentence_corpus_path)

    polling_thread = threading.Thread(
        target=poll,
        name="chatbot-backend",
        daemon=True,
    )
    polling_thread.start()
    try:
        yield
    finally:
        stop_polling.set()
        polling_thread.join(timeout=2)


app = FastAPI(title="Mineral Chatbot API", lifespan=lifespan)


def start_server():
    os.chdir(backend_directory)
    uvicorn.run(
        "server:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", "8000")),
    )


if __name__ == "__main__":
    start_server()
