FROM continuumio/miniconda3:4.5.11

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get -qq update \
    && apt-get install -yq --no-install-recommends \
      git \
      graphviz \
      libgraphviz-dev \
      pkg-config \
      tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DEPS="pip ffmpeg jupyter pytest pygraphviz numpy pandas pillow==2.9.0 requests scipy nltk six tqdm seaborn matplotlib pathos tensorflow contextlib2" \
    PIP_DEPS="python-magic coveralls pytest-cov pysrt xlrd clarifai pytesseract moviepy==0.2.2.13 SpeechRecognition IndicoIo sklearn python-twitter gensim oauth2client google-api-python-client google-compute-engine librosa ipython"
RUN conda info -a \
    && conda config --set always_yes yes --set changeps1 no \
    && conda config --add channels conda-forge \
    && conda update -y -q conda \
    && conda install $CONDA_DEPS \
    && conda clean --all --yes \
    && pip install --no-cache-dir $PIP_DEPS \
    && python -c 'import imageio; imageio.plugins.ffmpeg.download()'

WORKDIR /root/pliers
COPY . .
RUN python setup.py install \
    && python -m pliers.support.download

WORKDIR /work

CMD ["/bin/bash"]
