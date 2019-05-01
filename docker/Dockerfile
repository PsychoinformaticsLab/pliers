FROM python:3.6-slim
ARG DEBIAN_FRONTEND="noninteractive"
WORKDIR /opt/pliers
COPY . .
RUN apt-get update -qq \
    && tmp_pkgs="cmake gcc g++ libc6-dev libgraphviz-dev libmagic-dev make" \
    && apt-get install -yq --no-install-recommends \
        ffmpeg \
        graphviz \
        libmagic1 \
        tesseract-ocr \
        $tmp_pkgs \
    && pip install --no-cache-dir \
        --requirement requirements.txt \
        --requirement optional-dependencies.txt \
    && pip install --no-cache-dir --editable . \
    && apt-get autoremove --purge -yq $tmp_pkgs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --no-user-group --create-home --shell /bin/bash pliers
USER pliers
WORKDIR /work
