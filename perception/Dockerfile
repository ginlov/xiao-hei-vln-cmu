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

# --- runtime deps (SAM 2.1 + web layer + COCO RLE + OpenAI CLIP) ----------
# The base image already brings torch, ultralytics, opencv-python-headless,
# numpy and pillow. We only need what's missing. OpenAI's CLIP is the text
# encoder YOLOv8x-World v2 actually uses (the YAML default of mobileclip
# only kicks in for v1); pre-installing it stops Ultralytics from doing
# an `AutoUpdate` git-clone on the first /reload_classes call.
RUN pip install --no-cache-dir \
        sam2==1.1.0 \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.34.0" \
        python-multipart==0.0.20 \
        pycocotools==2.0.8 \
        git+https://github.com/ultralytics/CLIP.git

# --- bake in model weights so first /detect is fast ----------------------
# YOLOv8x-World v2 (~146 MB, largest open-vocab variant) and SAM 2.1
# Hiera Large (~900 MB — best mask quality; VRAM is not the constraint on
# the A10G and Task 1/2 answer within a 10-min budget, not real time).
RUN mkdir -p /opt/perception/models \
 && curl -fsSL -o /opt/perception/models/yolov8x-worldv2.pt \
        https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt \
 && curl -fsSL -o /opt/perception/models/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# --- bake in OpenAI CLIP ViT-B/32 weights for YOLO-World v2 ---------------
# YOLOv8x-World v2's saved checkpoint actually uses `clip:ViT-B/32`
# (not MobileCLIP, despite what the default-fallback model YAML
# suggests). Ultralytics' CLIP loader fetches it from
# openaipublic.azureedge.net on the first /reload_classes call;
# Azure throttles individual connections to ~200 KB/s, which means a
# cold call blocks ~30 minutes. We download host-side once (see
# commit message) and COPY it into the image at the path
# `ultralytics.utils.WEIGHTS_DIR / "clip"` so the loader finds it
# without going to the network.
RUN mkdir -p /ultralytics/weights/clip
COPY ViT-B-32.pt /ultralytics/weights/clip/ViT-B-32.pt

# Mark /opt/perception as a package so `from perception.geometry import ...`
# resolves; place application files under it.
RUN touch /opt/perception/__init__.py
COPY server.py pipeline.py geometry.py /opt/perception/

EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "perception.server:app", "--host", "0.0.0.0", "--port", "8001"]
