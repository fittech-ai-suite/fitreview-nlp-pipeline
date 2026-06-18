FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/roberta-fitness-binary/ ./models/roberta-fitness-binary/

ENV PYTHONUNBUFFERED=1
ENV ENV=production

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
