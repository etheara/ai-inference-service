FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY assessment_meta.joblib .
COPY assessment_model.joblib .

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}