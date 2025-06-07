# Gunakan image Python 3.12
FROM python:3.12-slim

# Atur direktori kerja di dalam container
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY app/requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy semua file project ke dalam container
COPY app/ .

# Expose port FastAPI
EXPOSE 8000

# Jalankan server via python main.py
CMD ["python", "main.py"]
