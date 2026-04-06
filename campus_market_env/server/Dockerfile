FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=7860

COPY . /app

RUN pip install --no-cache-dir -r campus_market_env/server/requirements.txt \
    && pip install --no-cache-dir .

EXPOSE 7860

CMD ["python", "main.py"]
