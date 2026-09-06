# semantic-search-foundations

Semantic search in ~40 lines of Python. No LangChain, no vector DB.

Documents and the query go through `all-MiniLM-L6-v2` to become vectors. Cosine
similarity ranks them. That's the whole thing.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then type a query. Asking "can I return an item?" returns the refund policy line,
even though they share no words.

## Files

- `documents.py` - the corpus, five hardcoded strings
- `embed.py` - loads the model, turns text into vectors
- `search.py` - cosine similarity, returns top 3
- `app.py` - the input loop
