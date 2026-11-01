FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/cargo
ENV RUSTUP_HOME=/rustup
ENV PYRE_SHARED_BUILD=/workspace/.pyre-build
ENV PATH=/cargo/bin:/workspace/.pyre-build/charon/linux-x86_64:$PATH
ENV CARGO_BUILD_JOBS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        file \
        git \
        libffi-dev \
        pkg-config \
        procps \
        python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable

COPY install-oracles.sh /usr/local/bin/pyre-install-oracles
RUN bash /usr/local/bin/pyre-install-oracles
ENV PYRE_CHECK_PYTHON3=/opt/pyre-oracles/bin/python3.15
ENV PYRE_CHECK_PYPY3=/opt/pyre-oracles/bin/pypy3

WORKDIR /workspace/pyre

CMD ["/bin/bash"]
