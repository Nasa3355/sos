FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libharfbuzz-dev \
    && apt-get clean

RUN pip install -r requirements.txt
# RUN pip install --upgrade setuptools==68.2.0 wheel

COPY client.py .

CMD ["python", "client.py"]