FROM python:3.11-slim

WORKDIR /app

# System deps OpenCV needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY api/ ./api/
COPY model/ ./model/
COPY frontend/ ./frontend/

# Create runtime directories
RUN mkdir -p /app/uploads /app/results

WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]