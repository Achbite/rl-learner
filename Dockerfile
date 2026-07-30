FROM python:3.11-slim

ARG TORCH_VERSION=2.12.1+cpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libabsl20240722 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libssl3t64 \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /opt/rl/learner
COPY . /opt/rl/learner
RUN cp -a _deps/contracts/python/. proto/ && \
    cp -a _deps/model-distributor /opt/rl/learner/model-distributor && \
    mkdir -p /opt/rl/identity && \
    cp _deps/identity/contracts.json /opt/rl/identity/contracts.json && \
    cp _deps/identity/model-distributor.json \
        /opt/rl/identity/model-distributor.json && \
    rm -rf _deps && \
    chmod +x run.sh scripts/entrypoint.sh && \
    chmod +x /opt/rl/learner/model-distributor/bin/maze_model_distributor && \
    python3 -m compileall -q main proto src tools

EXPOSE 9005 9200
ENTRYPOINT ["/opt/rl/learner/scripts/entrypoint.sh"]
