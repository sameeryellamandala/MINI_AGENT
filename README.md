<img width="1912" height="1027" alt="image" src="https://github.com/user-attachments/assets/670dca5d-9178-44f8-a5a9-2bd720c8ee6e" />


# Signal Search

Signal Search is a fast, persistent chat agent built for a one-tool challenge. Its only model-callable tool is Groq's built-in `browser_search`. It searches when facts need verification, cites sources, and remembers every question together with the time it was asked.

## Why it stands out

- **Exactly one tool:** `browser_search`
- **Fast:** Groq-hosted GPT-OSS with low reasoning effort
- **Grounded:** current web research with source links
- **Persistent:** LangGraph threads stored in SQLite
- **Time-aware:** every user question carries an ISO timestamp
- **Easy to run:** one Streamlit command and no external database

## Architecture

```text
Streamlit UI
     │
     ▼
LangGraph (one answer node) ─── SQLite checkpointer
     │
     ▼
Groq GPT-OSS
     │
     ▼
browser_search  ← the only model-callable tool
```

See [design.md](design.md) for the decisions and tradeoffs.

## Run locally

1. Create a free API key at [Groq Console](https://console.groq.com/keys).
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add the key:

   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

SQLite data is created at `data/agent.db` and is ignored by Git.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | Groq authentication |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model with browser-search support |
| `APP_TIMEZONE` | `Asia/Kolkata` | Timezone used for question timestamps |

For higher answer quality, set `GROQ_MODEL=openai/gpt-oss-120b`. The 20B default is friendlier to free-tier limits and optimized for response speed.

## Tests

```bash
pytest -q
```

Tests use a fake Groq client, so they require neither a key nor network access. They verify the single-tool boundary, timestamped memory, follow-up context, and thread ordering.

## Security

- Secrets are read from `.env` or Streamlit secrets and never committed.
- Web content is treated as untrusted evidence, not as executable instructions.
- The model receives no local files, shell, database, or code-execution tool.

## License

MIT

