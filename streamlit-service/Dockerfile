FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit слухає порт, який задає хост (локально 8501; на Railway/ін. платформах буде $PORT)
ENV CLIENT_API_URL=http://localhost:8000/ask_http \
    CLIENT_TIMEOUT=25 \
    APP_TITLE="Мій агент (Streamlit UI)" \
    PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["bash", "-lc", "streamlit run streamlit_service.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
