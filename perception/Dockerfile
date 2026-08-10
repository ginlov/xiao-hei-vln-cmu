# Build on Ultralytics' official image — it ships PyTorch + CUDA 12 +
# cuDNN + ultralytics + opencv-headless preinstalled, hosted on Docker
# Hub's fast CDN. We just layer SAM 2.1, FastAPI, and our code on top.
# Cuts the build from ~90 minutes (NVIDIA registry, throttled) to ~5
# minutes.
FROM ultralytics/ultralytics:8.4.72

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PERCEPTION_HOST=0.0.0.0 \
    PERCEPTION_PORT=8001 \
    # Make `perception` importable as a package (see /opt/perception/__init__.py).
    PYTHONPATH=/opt \
    # Cache locations under /opt so the bind mount over server.py/pipeline.py
    # doesn't shadow them.
    YOLO_CONFIG_DIR=/opt/perception/.ultralytics \
    HF_HOME=/opt/perception/.cache/huggingface \
    TORCH_HOME=/opt/perception/.cache/torch

WORKDIR /opt

# --- runtime deps (SAM 2.1 + OWLv2 + web layer + COCO RLE) ----------------
# The base image already brings torch, torchvision, opencv-python-headless,
# numpy and pillow. We only need what's missing. `transformers` provides the
# OWLv2 open-vocab detector (google/owlv2-large-patch14-ensemble), which
# replaced YOLO-World (and with it the OpenAI-CLIP text encoder).
RUN pip install --no-cache-dir \
        sam2==1.1.0 \
        transformers==4.46.3 \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.34.0" \
        python-multipart==0.0.20 \
        pycocotools==2.0.8

# --- bake in model weights so first /detect is fast ----------------------
# SAM 2.1 Hiera Large (~900 MB). YOLOv8x-World v2 was retired in favour of
# OWLv2; its weights (and the OpenAI-CLIP text encoder it needed) are gone.
RUN mkdir -p /opt/perception/models \
 && curl -fsSL -o /opt/perception/models/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# --- bake in OWLv2 weights so the first /detect doesn't hit the network ---
# google/owlv2-large-patch14-ensemble (~1.7 GB) → HF cache under HF_HOME.
# Downloading at build time keeps the first live tick fast and the runtime
# offline (no HuggingFace Hub call during the challenge).
RUN python3 -c "from transformers import Owlv2Processor, Owlv2ForObjectDetection; m='google/owlv2-large-patch14-ensemble'; Owlv2Processor.from_pretrained(m); Owlv2ForObjectDetection.from_pretrained(m)"

# Weights are now baked into the HF cache; forbid any Hub call at runtime so a
# live tick never blocks on the network. (Set AFTER the bake so the download
# above still runs.)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Mark /opt/perception as a package so `from perception.geometry import ...`
# resolves; place application files under it.
RUN touch /opt/perception/__init__.py
COPY server.py pipeline.py geometry.py /opt/perception/

EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "perception.server:app", "--host", "0.0.0.0", "--port", "8001"]
