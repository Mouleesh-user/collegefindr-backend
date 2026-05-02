# CollegeFindr Backend

Flask + SQLAlchemy API powering the CollegeFindr AI college search assistant.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # then fill in JWT_SECRET_KEY and OPENROUTER_API_KEY
python app.py
```

The server starts on `http://localhost:5000`.

## Environment variables

See `.env.example` for the full list. Required in production:

- `JWT_SECRET_KEY` — long random string (32+ chars)
- `OPENROUTER_API_KEY` — OpenRouter API key for the chat model
- `DATABASE_URL` — Postgres URL (falls back to local SQLite if unset)
- `ALLOWED_ORIGINS` — comma-separated frontend origins for CORS

## Tests

```bash
python tests/test_collegefindr.py
python tests/test_qa_runner.py
```

## Deploy

`render.yaml` defines the Render web service. Push to the branch Render watches; gunicorn is launched with the command in `render.yaml`.
