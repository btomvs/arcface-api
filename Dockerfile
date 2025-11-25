# Imagen base de Python
FROM python:3.11-slim

# Evitar .pyc y buffers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Carpeta de trabajo
WORKDIR /app

# Dependencias del sistema para opencv/onnx y curl
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear carpeta de modelos y descargar arcface.onnx desde HuggingFace
RUN mkdir -p models
RUN curl -L "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true" -o models/arcface.onnx

# Copiar el resto del código
COPY . .

# Puerto que usará Cloud Run
ENV PORT=8080

# Comando de arranque
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
