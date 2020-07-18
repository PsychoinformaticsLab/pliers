FROM python:3.7-slim as builder
ARG DEBIAN_FRONTEND="noninteractive"
WORKDIR /opt/pliers
COPY . .
# Install dependencies into a virtual environment so they can be easily copied into
# the second stage.
ENV PATH="/opt/venv/bin:$PATH"
RUN chmod a+rX -R . \
    && apt-get update -qq \
    && apt-get install -yq --no-install-recommends \
        cmake \
        gcc \
        g++ \
        libc6-dev \
        libgraphviz-dev \
        libmagic-dev \
        make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Run this in different layers for faster rebuilds when debugging. The extra layers
# do impose a size penalty, because we build the final image in a separate stage.
RUN python -m venv /opt/venv
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir --requirement requirements.txt
# Install the non-GPU tensorflow package because it is smaller.
RUN sed -i 's/tensorflow/tensorflow-cpu/g' optional-dependencies.txt \
    && python -m pip install --no-cache-dir --requirement optional-dependencies.txt
RUN python -m pip install --no-cache-dir --editable .
RUN python -m pip install --no-cache-dir ipython notebook
RUN python -m spacy download en_core_web_sm

FROM python:3.7-slim
RUN useradd --no-user-group --create-home --shell /bin/bash pliers \
    # Empty top level directories to facilitate use of the image in singularity
    # on a box with kernel lacking overlay FS support
    && mkdir -p /data /backup \
    && apt-get update -qq \
    && apt-get install -yq --no-install-recommends \
        ffmpeg \
        graphviz \
        libmagic1 \
        tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder --chown=pliers /opt/venv /opt/venv
COPY --from=builder --chown=pliers /opt/pliers /opt/pliers
ENV PATH="/opt/venv/bin:$PATH"
USER pliers
RUN python -m pliers.support.download \
    && python -m pliers.support.setup_yamnet
WORKDIR /work
