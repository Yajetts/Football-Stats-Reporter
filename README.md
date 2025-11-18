**Football Stats Reporter**

- **Type:** Streamlit RAG app (chat UI)
- **Purpose:** Answer football-only questions using a local knowledge base
- **Core stack:** `streamlit`, `llama-index`, `Gemini (Google)`, `Jina AI Embeddings`, `pydantic`, `python-dotenv`

**Overview**

- **What it does:**
	- Builds or loads a vector index over documents in `data/` and answers football-related questions via chat.
	- Uses a ReAct-style agent with retrieval to ground answers in your local knowledge base.
	- Maintains chat history per session and shows friendly error feedback.

- **What it won’t do:**
	- Answer non-football questions. The system prompt explicitly restricts scope to football stats.

**Features**

- **Chat UI with history:** Streamlit chat interface keeps `user`/`assistant` messages in session state.
- **Retrieval-Augmented Generation:** `llama_index` `VectorStoreIndex` over documents in `data/` powers grounded answers.
- **Gemini LLM:** Uses Google’s Gemini via `llama-index-llms-gemini` for responses.
- **Jina Embeddings:** Uses `jina-embeddings-v2-base-en` via `llama-index-embeddings-jinaai` for vectorization.
- **Scoped domain behavior:** System prompt limits answers to football stats only; other topics receive no response.
- **Friendly error handling:** Human-readable messages for model availability, authentication, and rate limits.
- **Session memory:** Lightweight `ChatMemoryBuffer` maintains recent turns for better context.
- **Index persistence:** Loads an existing index from `index/` if present; otherwise creates one from `data/`.
- **Extensible tooling (stubbed):** Code includes a template for adding custom tools (e.g., utility functions). The stub is not registered by default but shows how to extend the agent.

**Project Structure**

- `main.py`: Streamlit app entry. Handles UI, session state, and invoking the assistant.
- `utils/schema.py`: Assistant implementation: settings, index lifecycle, agent wiring, and query interface.
- `data/`: Your knowledge base. Place plain text (and other supported) files here. Sample: `test.txt` with dated goal events.
- `index/`: Persisted vector index (JSON store created by `llama_index`). Safe to delete to force a rebuild.
- `goalscorers.txt`: Large CSV of goal events (not auto-ingested unless moved under `data/`).
- `.env`: Environment variables (never commit real keys). See Setup.

**Prerequisites**

- Python 3.10+ recommended
- API keys:
	- `GEMINI_API_KEY` (Google Gemini)
	- `JINA_API_KEY` (Jina AI embeddings)
	- Optional (not used by default here): `LLAMA_CLOUD_API_KEY`, `PINECONE_API_KEY`

**Setup (Windows PowerShell)**

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install --upgrade pip
pip install streamlit pydantic python-dotenv \
		llama-index llama-index-core llama-index-llms-gemini llama-index-embeddings-jinaai
```

3) Create `.env` with your keys (do not share these)

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
JINA_API_KEY=your_jina_api_key_here
# Optional only if you later switch providers/integrations
# LLAMA_CLOUD_API_KEY=...
# PINECONE_API_KEY=...
```

4) Prepare data

- Put your football documents into `data/` (TXT works out of the box; LlamaIndex supports many formats).
- Example included: `data/test.txt` with dated scorers across matches.
- If you want to use `goalscorers.txt` (CSV), move or copy it under `data/` and consider adding a CSV-specific loader if you go beyond default text parsing.

**Run**

```powershell
streamlit run main.py
```

Then open the local URL printed by Streamlit (typically http://localhost:8501).

**Usage**

- Ask football-only questions in the chat input, for example:
	- “Who scored for Spain on June 5, 2022?”
	- “List the goals for Uruguay vs Bolivia on June 27, 2024.”
	- “Did Salomón Rondón score against Mexico in 2024?”
- The app retrieves relevant snippets from your `data/`-backed index and generates an answer via Gemini.
- A “Sources” expander is present in the UI; you can enhance `QueryResult` to return and display attributions per node.

**Configuration Details**

- LLM: `Gemini(model="gemini-pro")` configured via `GEMINI_API_KEY`.
- Embeddings: `JinaEmbedding(model="jina-embeddings-v2-base-en")` via `JINA_API_KEY`.
- Retrieval: `VectorStoreIndex` from `llama_index.core` with `similarity_top_k=10` in the query engine.
- Memory: `ChatMemoryBuffer` with a token limit of 4096.
- Domain guardrails: System prompt restricts to football-only responses.

**Index Lifecycle**

- On first run with no `index/`, documents from `data/` are ingested and an index is built.
- With an existing `index/`, the app loads the stored index and skips ingestion.
- To force a rebuild after changing `data/`, delete the `index/` directory and re-run the app.
- Note: `save_index()` in `utils/schema.py` is currently a placeholder; the repo includes a prebuilt `index/`. If you remove it and expect persistence across runs, implement `save_index()` using `self.index.storage_context.persist(persist_dir=self.index_path)`.

**Extending the Assistant**

- Add more tools: `utils/schema.py` includes a stubbed `custom_tool`. To enable, pass it into `ReActAgent.from_tools([...])` alongside the retrieval tool.
- Richer sources: Change `QueryResult.source_nodes` to carry original `NodeWithScore` objects (or strings you format) and render them in `main.py`.
- Additional loaders: Use LlamaIndex loaders for CSV, PDFs, web pages, etc., to broaden your `data/` ingestion.

**Troubleshooting**

- “Authentication failed”: Check `.env` and ensure the PowerShell session loaded the correct environment.
- “The requested model is not available”: Verify the Gemini model name or your account access.
- “Rate limit exceeded”: Wait and retry; consider lowering request frequency.
- Blank/odd answers: Ensure your `data/` actually contains content relevant to the question; rebuild the index if you changed files.
- Import errors: Ensure all dependencies are installed in the active virtual environment.

**Security Notes**

- Never commit your real `.env` values. Treat API keys as secrets.
- If you checked in `.env` accidentally, rotate the keys immediately and purge history if needed.

**FAQ**

- Can it answer general sports or non-sports questions?
	- No. It is intentionally scoped to football. Adjust the system prompt in `utils/schema.py` to change behavior.
- How do I add CSV data like `goalscorers.txt`?
	- Move/copy under `data/` and optionally implement a CSV loader for structured parsing; otherwise, plain text ingestion works but loses schema.
- How do I reset the chat?
	- Refresh the browser or restart the app to clear Streamlit session state.

