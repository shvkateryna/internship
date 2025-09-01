FROM python:3.11-slim
WORKDIR /app
COPY . .
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8003
CMD ["python", "telegram_service.py"]
