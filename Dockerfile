# Python base image (slim = smaller download, fewer vulnerabilities)
FROM python:3.12-slim

# Print logs immediately and don't write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Install dependencies first: this layer is cached and only rebuilt
# when requirements.txt changes, which makes rebuilds much faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the trained model
COPY app/ ./app/
COPY model/ ./model/

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# 0.0.0.0 = listen on all interfaces so traffic from outside the container reaches the app.
# Render injects a PORT environment variable; locally we fall back to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
