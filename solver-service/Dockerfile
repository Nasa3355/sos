FROM python:3.12-slim

WORKDIR /app  

COPY requirements.txt .  
COPY solver_service.py .

RUN apt-get update && apt-get install -y \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libharfbuzz-dev \
    && apt-get clean

# RUN pip install --upgrade setuptools==68.2.0 wheel
RUN pip install -r requirements.txt 

CMD ["uvicorn", "solver_service:app", "--host", "0.0.0.0", "--port", "8002"]