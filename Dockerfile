FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY app.py train.py ./

# Train the model at build time
RUN python train.py

EXPOSE 5000
CMD ["python", "app.py"]
