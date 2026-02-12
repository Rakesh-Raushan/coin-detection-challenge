#Official lightweight python image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

#dependencies for openCV
# RUN apt-get update && apt-get install -y \
# libgl1-mesa-glx \
# libglib2.0-0 \
# && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#Creating non root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/weights /code/app/weights

#copy app code
COPY app/ ./app/

#Change ownership of code to non-root user
#creating data directory here so that user has write permissions
RUN mkdir -p data/uploads && chown -R appuser:appuser /code

#switch to non root
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

