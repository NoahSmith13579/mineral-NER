# Mineral NER

A machine learning project that uses **Named Entity Recognition (NER)** and **TF-IDF information retrieval** to identify mineral-related concepts and answer questions about mineral properties, classifications, and applications.

The project uses [spaCy](https://spacy.io/) to train a custom NLP pipeline that recognizes mineral-related entities. It combines this model with a sentence retrieval system to identify relevant information from a mineral knowledge corpus and return context-based answers.

## Frontend Availability

**Important:** This repository contains the machine learning and backend components of the original chatbot project. The original frontend was developed using copyrighted material that I do not own or have permission to redistribute. Consequently, the frontend source code is not included in this repository.
The backend was designed to communicate with the original frontend through a REST API.
This repository focuses on demonstrating the NLP, named entity recognition, model training, and information retrieval components independently of the original frontend.

## Features

- **Custom Named Entity Recognition:** Identifies mineral names, classifications, properties, and uses.
- **NLP Model Training:** Trains and evaluates custom spaCy NER models using annotated mineral data.
- **Hyperparameter Optimization:** Uses grid search to compare dropout, learning rate, and training step configurations.
- **TF-IDF Information Retrieval:** Uses TF-IDF vectorization and cosine similarity to identify relevant sentences.
- **Lemmatization:** Uses spaCy's lemmatizer to normalize terms during information retrieval.
- **REST API:** Provides a FastAPI backend for processing questions and communicating with an external chatbot frontend.
- **Rule-Based Responses:** Handles certain predefined questions without requiring model inference.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Core application and machine learning implementation |
| spaCy | NLP pipeline and named entity recognition |
| scikit-learn | TF-IDF vectorization and cosine similarity |
| Hugging Face Datasets | Loading and processing the source dataset |
| NumPy | Numerical operations |
| FastAPI | Backend API |
| Uvicorn | API server |
| Requests | Frontend communication |

## Named Entity Recognition

The custom NER model identifies four categories of mineral-related entities.

| Entity | Description | Example |
|---|---|---|
| `MINERAL_NAME` | Names of individual minerals | Quartz, Hematite, Indium |
| `MINERAL_GROUP` | Mineral classifications and groups | Silicates, Oxides |
| `MINERAL_PROPERTY` | Physical and chemical properties | Hardness, Crystal System |
| `MINERAL_USE` | Applications and uses of minerals | Jewelry, Electronics |

These entities provide additional context for understanding questions and retrieving relevant information.

## Architecture

The project follows a pipeline consisting of corpus generation, model training, entity extraction, information retrieval, and response generation.

### 1. Corpus Generation

The corpus builder loads mineral-related articles from a Wikipedia dataset in Parquet format.

- Filters articles using a predefined list of mineral names.
- Processes article text and divides it into sentences.
- Annotates mineral names, groups, properties, and uses.
- Generates training, validation, and testing datasets.

The datasets are saved in spaCy's `.spacy` binary format.

### 2. Model Training

A custom spaCy NLP pipeline is trained using the generated annotations.

The pipeline includes:
- `tok2vec` – Generates contextual token representations.
- `ner` – Identifies and classifies named entities.
- `lemmatizer` – Normalizes words to their base forms.

Hyperparameter experiments compare different combinations of:
- Dropout rates
- Learning rates
- Training steps

Models are evaluated using entity precision, recall, and F1 score. The best-performing model is selected for the chatbot.

### 3. Entity Extraction

When a question is submitted, the trained NLP pipeline identifies relevant mineral-related entities.

For example:

**Question:** Is quartz used in jewelry?

**Identified entities:**
- `MINERAL_NAME`: Quartz
- `MINERAL_USE`: Jewelry

The extracted entities help narrow the search to information about the specified mineral and its relevant characteristics.

### 4. Information Retrieval

The retrieval system uses TF-IDF vectorization and cosine similarity to identify the most relevant sentence.

The process:

1. Filters the sentence corpus to articles matching the identified mineral.
2. Extracts relevant entity terms from the question.
3. Lemmatizes and normalizes the question and candidate sentences.
4. Calculates TF-IDF vectors.
5. Compares the question against candidate sentences using cosine similarity.
6. Returns the highest-scoring sentence if it exceeds the similarity threshold.

This approach provides answers grounded in the source corpus rather than generating new information.

### 5. Backend Communication

FastAPI hosts the chatbot backend and manages communication with the original frontend.

The backend:
- Loads the trained spaCy model.
- Loads the mineral sentence corpus.
- Retrieves questions from the frontend.
- Processes questions using the chatbot response pipeline.
- Returns answers to the frontend.

The backend exposes REST API endpoints for frontend communication. The original frontend implementation is not included in this repository.

## Project Structure

```text
mineral-NER/
│
├── chatbot/
│   └── responses.py
│
├── config/
│   └── settings.py
│
├── corpus/
│   ├── annotations.py
│   ├── builder.py
│   └── corpus_io.py
│
├── retrieval/
│   └── sentence_retrieval.py
│
├── training/
│   ├── evaluation.py
│   └── grid_search.py
│
├── main.py
├── server.py
├── config.cfg
├── minerals.py
├── terms.py
├── requirements.txt
└── README.md
```

Generated datasets, trained models, and external source data are maintained separately from the source code.

## Installation and Setup

### Prerequisites

- Python
- pip
- Git
- Required mineral source dataset and preprocessing pipeline

### 1. Clone the Repository

```bash
git clone https://github.com/NoahSmith13579/mineral-NER.git

cd mineral-NER
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Configure the required environment variables and data paths for your local environment.

For example:

```env
BASE_PATH=path/to/project/data
BACKEND_PATH=path/to/mineral-NER
FRONTEND_URL=http://localhost:8080
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
```

Adjust the paths and settings to match your local environment.

### 4. Prepare the Source Data

The application expects external resources, including:

- Wikipedia article data in Parquet format under `Data/Wikipedia/`.
- A preprocessed spaCy pipeline at `Data/Module-7/minerals_tokenized.pipeline`.

These resources are required by the corpus generation process and are not included in the repository.

### 5. Running the Application

The FastAPI backend is designed to communicate with the original chatbot frontend. Since the frontend source code is not distributed with this repository, the complete original chatbot interface cannot be reproduced using this repository alone.

The machine learning, NER, and sentence retrieval components can be examined independently of the frontend. Running the complete application requires the original frontend and its associated configuration.

## Limitations

- The knowledge base is limited to the mineral articles available in the source dataset.
- The system relies on its trained NER model to identify relevant concepts.
- TF-IDF retrieval depends on lexical similarity and may struggle with questions that use unfamiliar terminology or substantially different wording.
- Responses are limited to information available in the sentence corpus.
- The system does not use a generative large language model (LLM) to formulate answers.
- The original chatbot frontend is not included because it contains copyrighted material that cannot be redistributed.
- External datasets and trained model artifacts must be obtained or generated separately.

## Future Improvements

Potential areas for improvement include:

- Expanding the mineral knowledge base with additional scientific sources.
- Improving entity recognition through additional annotated training data.
- Experimenting with alternative information retrieval techniques.
- Improving semantic matching for questions that use synonyms or unfamiliar terminology.
- Evaluating retrieval accuracy independently from NER performance.
- Expanding the range of supported questions and mineral-related concepts.
- Improving backend response handling and API documentation.

## Author

**Noah Smith**

[GitHub Profile](https://github.com/NoahSmith13579)

---

*This project was developed as an exploration of natural language processing, named entity recognition, machine learning, and information retrieval.*
