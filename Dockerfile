FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=7860

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .
COPY __init__.py .
COPY client.py .
COPY config.py .
COPY enums.py .
COPY gym_env.py .
COPY models.py .
COPY openenv.yaml .
COPY server ./server
RUN pip install --no-cache-dir -r requirements.txt

COPY docs ./docs
COPY static ./static
COPY tasks ./tasks
COPY run_agent.py .
COPY test_env.py .
COPY inference.py .
COPY main.py .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860

CMD ["python", "main.py"]
