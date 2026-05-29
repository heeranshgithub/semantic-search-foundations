from documents import DOCUMENTS
from embed import embed_text
from search import search

# Embed documents here
doc_embeddings = embed_text(DOCUMENTS)

while True:
    query = input("\nEnter query: ")

    query_embedding = embed_text([query])[0]

    results = search(query_embedding, doc_embeddings, DOCUMENTS)

    print("\nTop Matches:\n")

    for doc, score in results:
        print(f"{score:.4f} | {doc}")