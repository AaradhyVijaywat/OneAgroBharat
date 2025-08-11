FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8080
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
CMD ["python", "app.py"]
