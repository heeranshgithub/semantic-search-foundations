# Phase 1 Minimal Project — Build Semantic Search From Scratch

This is the perfect Phase 1 project because it teaches:

```text id="yz1uz0"
language → vectors → similarity → retrieval
```

without hiding anything behind frameworks.

No:

* LangChain
* vector DB
* orchestration
* magic abstractions

Just first principles.

---

# Project Goal

Build a tiny semantic retrieval engine in Python.

User types:

```text id="vbnffq"
"can I return an item?"
```

System retrieves:

```text id="1e6zjz"
"Our refund policy allows returns within 30 days."
```

even though the wording differs.

That’s the core of RAG.

---

# What You Will Build

```text id="2q8l5j"
documents
→ embeddings
→ vector similarity
→ ranked retrieval
```

No LLM yet.

Pure retrieval intuition.

---

# Tech Stack

Keep it brutally minimal.

Use:

* Python
* NumPy
* sentence-transformers

That’s it.

---

# Why sentence-transformers?

Because you want to learn:

* retrieval mechanics
  not:
* train embedding models from scratch

We’ll use pretrained embeddings as a microscope.

Good starter model:

```text id="7w5v8r"
all-MiniLM-L6-v2
```

from Hugging Face + SentenceTransformers ecosystem.

---

# PROJECT STRUCTURE

```text id="yvlqoj"
phase1_semantic_search/
│
├── app.py
├── documents.py
├── embed.py
├── search.py
├── requirements.txt
└── notes.md
```

---

# STEP 1 — Setup Environment

Create project:

```bash
mkdir phase1_semantic_search
cd phase1_semantic_search
```

Create venv:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install sentence-transformers numpy
```

---

# STEP 2 — Create Tiny Document Corpus

`documents.py`

```python
DOCUMENTS = [
    "Our refund policy allows returns within 30 days.",
    "Shipping takes 3 to 5 business days.",
    "Reset your password using the settings page.",
    "Dogs are loyal domestic animals.",
    "Kubernetes manages containerized applications.",
]
```

---

# Why This Matters

You are building:

* semantic space

Every sentence becomes:

* a point in vector space

---

# STEP 3 — Generate Embeddings

`embed.py`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    return model.encode(texts)
```

---

# What’s Happening Internally

This line:

```python
model.encode(texts)
```

does:

```text id="84qubq"
text
→ tokens
→ transformer
→ embedding vector
```

You are literally generating semantic coordinates.

---

# STEP 4 — Build Similarity Search

`search.py`

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_embedding, doc_embeddings, documents, top_k=3):
    scores = []

    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((documents[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]
```

---

# Critical Learning Here

THIS is retrieval.

Not Pinecone.
Not LangChain.

This loop:

```python
for each document:
    compare similarity
    rank results
```

is the beating heart of semantic search.

---

# STEP 5 — Wire Everything Together

`app.py`

```python
from documents import DOCUMENTS
from embed import embed_text
from search import search

# Embed documents once
doc_embeddings = embed_text(DOCUMENTS)

while True:
    query = input("\nEnter query: ")

    query_embedding = embed_text([query])[0]

    results = search(
        query_embedding,
        doc_embeddings,
        DOCUMENTS
    )

    print("\nTop Matches:\n")

    for doc, score in results:
        print(f"{score:.4f} | {doc}")
```

Run:

```bash
python app.py
```

---

# Example Queries

Try:

```text id="4o7dph"
How do returns work?
```

Expected retrieval:

* refund policy

---

Try:

```text id="zh8yr7"
pet animal
```

Expected retrieval:

* dogs are loyal domestic animals

---

Try:

```text id="h5s5zj"
containers orchestration
```

Expected retrieval:

* Kubernetes sentence

---

# This Is the Key Moment

You’ll notice:

```text id="m1nyhc"
semantic matching without exact keywords
```

That’s the conceptual breakthrough.

---

# STEP 6 — Add Visibility Into the Geometry

Print embedding info.

Example:

```python
print(query_embedding.shape)
```

You’ll see something like:

```text id="f8xj5j"
(384,)
```

Meaning:

* each sentence = point in 384-dimensional space

---

# STEP 7 — Observe Similarity Scores

Print all scores.

You’ll start seeing:

* strong semantic neighbors
* weak neighbors
* accidental neighbors

This teaches:

* embedding imperfections
* ranking behavior

---

# STEP 8 — Manual Exploration Exercises

These matter a LOT.

---

## Exercise 1 — Synonyms

Query:

```text id="jlwm3k"
refund
return
money back
```

Observe:

* semantic closeness

---

## Exercise 2 — Ambiguity

Add docs like:

```python
"Python is a programming language."
"Python is a large snake."
```

Query:

```text id="8i3x2m"
python
```

Observe ambiguity.

---

## Exercise 3 — Failure Cases

Query nonsense:

```text id="q4p1v0"
banana spaceship democracy
```

Observe weird neighbors.

This teaches:

* embedding noise
* latent structure artifacts

---

# STEP 9 — Build Intuition About Dimensions

You are NOT supposed to interpret dimensions directly.

Wrong mental model:

```text id="0c3cl4"
dimension 12 = happiness
```

Correct model:

```text id="j6fymr"
meaning is distributed across many dimensions
```

---

# STEP 10 — Add Top-K Retrieval

Experiment:

* top 1
* top 3
* top 5

Observe:

* precision vs recall

This becomes important later in RAG.

---

# STEP 11 — Build a Tiny Mental Model

You should now mentally see:

```text id="2z9pk3"
documents floating in semantic space
```

Query:

* becomes another vector
* nearest neighbors retrieved

That intuition is priceless.

---

# STEP 12 — Reflection Questions

After finishing, you should answer:

---

## Q1

Why can:

```text id="6l4m0p"
"return item"
```

retrieve:

```text id="cq7gf8"
"refund policy"
```

without keywords?

---

## Q2

Why does cosine similarity work?

---

## Q3

Why are embeddings geometric representations?

---

## Q4

Why might retrieval fail?

---

## Q5

Why does chunk granularity matter?

---

# What NOT To Do Yet

Avoid:

* LangChain
* vector DBs
* agents
* memory systems
* fancy orchestration

Right now you are learning:

* the primitive itself

---

# What You’ll Understand After This

You’ll deeply grok:

* embeddings
* semantic space
* vector similarity
* ranking
* retrieval mechanics

And THEN:

* vector DBs become obvious
* RAG pipelines feel simple
* ANN search makes sense

---

# Best Next Step After Completing This

After this project:

```text id="bzqpf6"
Phase 1.5 → Build your own chunking experiments
```

Because chunking is where retrieval quality becomes real.
