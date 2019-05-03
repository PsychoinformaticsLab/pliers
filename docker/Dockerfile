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
        ipython \
        notebook \
    && pip install --no-cache-dir --editable . \
    && python -m spacy download en_core_web_sm \
    && rm -rf ~/.cache/pip \
    && apt-get autoremove --purge -yq $tmp_pkgs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --no-user-group --create-home --shell /bin/bash pliers
USER pliers
RUN python -m pliers.support.download
WORKDIR /work
