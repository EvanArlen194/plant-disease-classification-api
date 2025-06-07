FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 wget && \
    pip install --upgrade pip && \
    pip install gdown && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/ .

RUN pip install -r requirements.txt

RUN echo "App structure:" && find . -name "*.keras" -o -name "*.model" || true

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]
