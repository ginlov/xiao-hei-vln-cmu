FROM zhangjicmu/ubuntu24_ros:ai_module

# The base image ships ROS Jazzy but not pip. Add it once at the apt layer so
# subsequent rebuilds skip this when nothing changed.
USER root
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-pip \
 && rm -rf /var/lib/apt/lists/*

# --- our package ------------------------------------------------------------
WORKDIR /opt/xiao_hei_vln
COPY pyproject.toml README.md ./
COPY src ./src

# Install into the same Python rclpy lives in (system Python 3.12). Ubuntu 24.04
# marks the system Python as externally-managed; --break-system-packages is the
# accepted escape hatch inside an isolated container image. Editable mode lets a
# bind mount over /opt/xiao_hei_vln/src take effect without reinstalling.
#
# Build arg `XIAO_HEI_EXTRA` controls which optional dep group is installed:
#   - "" (default): no extras → dummy responder only. Fast build.
#   - "qwen":       installs `openai` + `pillow` for the HTTP sidecar engine
#                   (lightweight, no CUDA conflicts).
#   - "qwen-local": installs vLLM + torch + CUDA wheels in-process (heavy,
#                   may conflict with apt packages — not recommended).
ARG XIAO_HEI_EXTRA=""
RUN if [ -n "$XIAO_HEI_EXTRA" ]; then \
        python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed -e ".[${XIAO_HEI_EXTRA}]"; \
    else \
        python3 -m pip install --no-cache-dir --break-system-packages -e .; \
    fi

# --- ROS env ---------------------------------------------------------------
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/docker/.bashrc \
 && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/docker/.bashrc

# --- entrypoint ------------------------------------------------------------
COPY docker/entrypoint.sh /usr/local/bin/xiao_hei_entrypoint.sh
RUN chmod +x /usr/local/bin/xiao_hei_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/xiao_hei_entrypoint.sh"]
CMD ["xiao-hei-vlm"]
