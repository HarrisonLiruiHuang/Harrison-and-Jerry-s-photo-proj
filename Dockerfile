FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/checkpoints/torch_cache && chmod -R 777 /app/checkpoints
ENV TORCH_HOME=/app/checkpoints/torch_cache

EXPOSE 7860
ENV PORT=7860

CMD python web_app.py --host 0.0.0.0 --port ${PORT}
