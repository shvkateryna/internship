# Tasia-Perekladasia — Tool-Using AI Agent (RAG + Translation)

A small, containerized AI system that:
1. Answers questions about me via a tiny RAG service (FAISS over a local text file), and
2. Performs English → Ukrainian translations via a local seq2seq model,
3. Exposes a Streamlit chat UI and a Telegram bot,
4. Orchestrates tool-calling and chat memory with LangChain + Redis.

## Architecture

<img width="803" height="347" alt="Screenshot 2025-09-01 at 23 29 43" src="https://github.com/user-attachments/assets/3eb6be3e-4119-4c38-a383-3e1b293bdcae" />

## Services

### Client service
Purpose: Central LangChain agent (ChatOpenAI) that chooses tools and keeps chat history in Redis.

Key features:
1. Strict routing rules.
2. Memory: RedisChatMessageHistory with TTL (configurable).
3. MCP tool exposed: ask(input, session_id) — runs the agent with session-scoped memory.

### RAG service
Purpose: Answers personal questions from a tiny local RAG index.

Tools:
1. about_me_search(question) — returns an answer strictly from context or “No data available.”
2. rag_reindex() — rebuilds embeddings/index from the data file.

### Translation service
Purpose: Deterministic English → Ukrainian translation using a local mt5 pretrained model. (BLEU 17.7, chrf 43).

Tools:
1. translate(user_input, language)

### Streamlit service
Purpose: Simple web chat with avatars and a pleasant theme.

Talks to client-service via MCP, shows a running dialog, and keeps a session-scoped session_id.

### Telegram service
Purpose: Telegram bot gateway.

Endpoints:
1. POST /telegram/webhook — Telegram webhook.
2. GET /health — healthcheck.

## How to run the project:
1. docker compose build --no-cache
2. docker compose up -d

To close the project run: docker compose down -v.
