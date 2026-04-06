FROM python:3.10-slim

WORKDIR /app

# Install all necessary dependencies deterministically
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy logic structure
COPY . .

# Generate background claims, forcefully succeeding if environment errors or skips occur
RUN python data_gen.py || true

EXPOSE 7860

# Secure invocation binding properly mapped to network boundaries
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
