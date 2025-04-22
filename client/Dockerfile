FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --upgrade setuptools==68.2.0 wheel

COPY client.py .

CMD ["python", "client.py"]