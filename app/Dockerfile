FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 wget curl && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/api/keras_model

RUN curl -L -o /app/api/keras_model/best_model.h5 https://models.evanarlen.my.id/best_model.h5

COPY app/ .

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]
