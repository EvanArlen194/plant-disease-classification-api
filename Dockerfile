# Gunakan image Python 3.12
FROM python:3.12-slim

# Install library sistem yang dibutuhkan cv2
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Atur direktori kerja
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY app/requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project ke dalam container
COPY app/ .

# Expose port FastAPI
EXPOSE 8000

# Jalankan server via python main.py
CMD ["python", "main.py"]
