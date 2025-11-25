FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencias del sistema para opencv/onnx y curl
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Carpeta de modelos + descarga modelo arcface
RUN mkdir -p models
RUN curl -L "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" \
    -o models/arcface.onnx

# Copiar todo el código
COPY . .

# Cloud Run pone PORT, pero dejamos 8080 por defecto
ENV PORT=8080

# Punto de entrada: nuestro script de arranque
CMD ["python", "start.py"]
