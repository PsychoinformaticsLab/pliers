FROM python:3.7-slim as builder
ARG DEBIAN_FRONTEND="noninteractive"
WORKDIR /opt/pliers
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
        libsndfile1 \
        make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Run this in different layers for faster rebuilds when debugging. The extra layers
# do impose a size penalty, because we build the final image in a separate stage.
RUN python -m venv /opt/venv
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir ipython notebook
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --requirement requirements.txt
# Install the non-GPU tensorflow package because it is smaller.
COPY optional-dependencies.txt .
RUN sed -i 's/tensorflow>/tensorflow-cpu>/g' optional-dependencies.txt \
    && python -m pip install --no-cache-dir --requirement optional-dependencies.txt
COPY . .
RUN python -m pip install --no-cache-dir --editable .
RUN python -m spacy download en_core_web_sm
RUN python -m pliers.support.download \
    && python -m pliers.support.setup_yamnet
LABEL org.opencontainers.image.source=https://github.com/PsychoinformaticsLab/pliers

FROM python:3.7-slim as base
# Empty top level directories to facilitate use of the image in singularity
# on a box with kernel lacking overlay FS support
RUN mkdir -p /data /backup \
    && apt-get update -qq \
    && apt-get install -yq --no-install-recommends \
        ffmpeg \
        graphviz \
        libmagic1 \
        tesseract-ocr \
        make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
LABEL org.opencontainers.image.source=https://github.com/PsychoinformaticsLab/pliers

FROM base as executable
RUN useradd --create-home --shell /bin/bash pliers
COPY --from=builder --chown=pliers /opt/venv /opt/venv
COPY --from=builder --chown=pliers /opt/pliers /opt/pliers
COPY --from=builder --chown=pliers /root/nltk_data /home/pliers/nltk_data
COPY --from=builder --chown=pliers /root/pliers_data /home/pliers/pliers_data
ENV PATH="/opt/venv/bin:$PATH"
USER pliers
WORKDIR /work
LABEL org.opencontainers.image.source=https://github.com/PsychoinformaticsLab/pliers

FROM executable as docs
WORKDIR /opt/pliers/docs
RUN python -m pip install /opt/pliers['docs']
LABEL org.opencontainers.image.source=https://github.com/PsychoinformaticsLab/pliers
CMD make doctest && make html


FROM executable as tests
WORKDIR /opt/pliers
RUN python -m pip install /opt/pliers['tests']
ENV skip_high_memory true
LABEL org.opencontainers.image.source=https://github.com/PsychoinformaticsLab/pliers
CMD py.test pliers/tests -n auto --cov=pliers --cov-report xml -m "not requires_payment" -W ignore::UserWarning