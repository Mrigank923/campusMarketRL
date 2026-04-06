FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=7860

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir -r requirements.txt

COPY campus_market_env ./campus_market_env
COPY docs ./docs
COPY static ./static
COPY main.py .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860

CMD ["python", "main.py"]
