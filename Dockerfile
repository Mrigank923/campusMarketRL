FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

COPY campus_market_env ./campus_market_env
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir .

COPY --from=frontend-builder /app/frontend/out ./static

COPY campus_market_env/server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "campus_market_env.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
