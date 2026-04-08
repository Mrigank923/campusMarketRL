FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=7860

COPY . /app

RUN pip install --no-cache-dir -r server/requirements.txt \
    && pip install --no-cache-dir .

EXPOSE 7860

CMD ["python", "main.py"]
