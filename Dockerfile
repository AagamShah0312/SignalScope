# =============================================================================
# SignalScope API — Docker image
# =============================================================================
# Build:  docker build -t signalscope-api .
# Run:    docker run -p 8000:8000 signalscope-api
#
# The shipped checkpoint (src/models/best_efficientnet_b0.pth) is copied into
# the image, so a fresh `docker run` can serve predictions immediately.
#
# For a smaller CPU-only image, replace the single pip line with:
#   RUN pip install --no-cache-dir torch torchvision \
#       --index-url https://download.pytorch.org/whl/cpu \
#       && pip install --no-cache-dir -r requirements.txt
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# opencv-headless needs a couple of shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY config.yaml .
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
