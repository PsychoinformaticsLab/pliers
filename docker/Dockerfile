FROM ubuntu:trusty

RUN apt-get update && apt-get install -y git
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:joyard-nicolas/ffmpeg -y
RUN apt-get -qq update
RUN apt-get install -y ffmpeg tesseract-ocr
RUN apt-get install -y wget
RUN apt-get install -y graphviz libgraphviz-dev pkg-config

ENV MINICONDA_URL "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ENV CONDA_DEPS "pip jupyter pytest numpy pandas pillow==2.9.0 requests scipy nltk six tqdm seaborn matplotlib pathos tensorflow contextlib2"
WORKDIR /root
ENV MINICONDA $HOME/miniconda
ENV PATH "$MINICONDA/bin:$PATH"
RUN wget $MINICONDA_URL -O miniconda.sh;
RUN bash miniconda.sh -b -f -p $MINICONDA;
RUN hash -r
RUN conda info -a
RUN conda config --set always_yes yes --set changeps1 no
RUN conda config --add channels conda-forge
RUN conda update -y conda
RUN conda create -y -n py35 python=3.5 $CONDA_DEPS

RUN /bin/bash -c "source /miniconda/envs/py35/bin/activate py35 \
  && pip install --upgrade --ignore-installed setuptools \
  && pip install python-magic coveralls pytest-cov pygraphviz pysrt xlrd clarifai pytesseract moviepy==0.2.2.13 SpeechRecognition IndicoIo sklearn python-twitter gensim oauth2client google-api-python-client google-compute-engine librosa ipython"

RUN /bin/bash -c 'source /miniconda/envs/py35/bin/activate py35 \
 && python -c "import imageio; imageio.plugins.ffmpeg.download()"'

RUN git clone https://github.com/tyarkoni/pliers.git
WORKDIR /root/pliers
RUN /bin/bash -c 'source /miniconda/envs/py35/bin/activate py35 \
 && python setup.py install'
RUN echo "if (tty -s); then \n\
    source /miniconda/envs/py35/bin/activate py35\n\
fi" >> /root/.bashrc
RUN /bin/bash -c 'source /miniconda/envs/py35/bin/activate py35 \
 && python -m pliers.support.download'
RUN apt-get install -y curl
CMD ["/bin/bash"]
