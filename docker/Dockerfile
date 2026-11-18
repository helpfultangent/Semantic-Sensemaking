# One image for every Semantic Sensemaking Tapis app; the app definition picks
# the workflow with SSM_WORKFLOW. Build from the repository root:
#   docker build -f docker/Dockerfile -t semantic-sensemaking .
FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yaml /tmp/environment.yaml
RUN micromamba install -y -n base -f /tmp/environment.yaml && \
    micromamba clean --all --yes

# Run the following RUN steps inside the activated conda base environment.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /opt/semantic-sensemaking
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src
# Installed, not copied onto sys.path: the image runs exactly what `pip install` gives users.
# [embed] (PyTorch) is deliberately excluded; on TACC, serve models with vLLM or Ollama and
# use the openai_compatible backend.
RUN python -m pip install --no-cache-dir ".[tacc]" && semantic-sensemaking --version

COPY --chmod=755 docker/run.sh /tapis/run.sh

# Apptainer images are read-only and $HOME may not be writable on compute nodes.
ENV PATH="/opt/conda/bin:${PATH}" \
    XDG_CACHE_HOME=/tmp/ssm-cache \
    HF_HOME=/tmp/ssm-cache/huggingface \
    MPLCONFIGDIR=/tmp/ssm-cache/matplotlib

WORKDIR /tmp
ENTRYPOINT ["/tapis/run.sh"]
