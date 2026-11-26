FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements-video.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-video.txt
COPY traffic_trajectory/ ./traffic_trajectory/
USER 10001:10001

ENTRYPOINT ["python", "-m", "traffic_trajectory"]
