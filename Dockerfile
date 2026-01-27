FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "200"]
