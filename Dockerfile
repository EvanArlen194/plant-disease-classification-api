FROM python:3.12-slim
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/ .
COPY keras_model ./keras_model

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN echo "App structure:" && find . -name "*.keras" -o -name "*.model"

EXPOSE 8000
CMD ["python", "main.py"]
