FROM python:3.12-slim

WORKDIR /app

COPY gateway.py .
COPY requirements.txt .

RUN pip install --upgrade setuptools==68.2.0 wheel
RUN pip install -r requirements.txt

CMD ["uvicorn", "solver_service:app", "--host", "0.0.0.0", "--port", "8002"]