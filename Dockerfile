FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    curl \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:${PATH}
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
RUN rustup target add wasm32-wasip1
RUN cargo install wasm-tools

WORKDIR /opt/ferrite
COPY . .

RUN cargo build -p ferrite-cli --release
RUN cargo build -p mistral-inference --target wasm32-wasip1 --release
RUN wasm-tools component embed wit \
    target/wasm32-wasip1/release/mistral_inference.wasm \
    -o target/wasm32-wasip1/release/mistral_inference.embed.wasm
RUN wasm-tools component new \
    target/wasm32-wasip1/release/mistral_inference.embed.wasm \
    --adapt adapters/wasi_snapshot_preview1.reactor.wasm \
    -o /opt/ferrite/mistral_inference.component.wasm

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ferrite
COPY --from=builder /opt/ferrite/target/release/ferrite-rt /usr/local/bin/ferrite-rt
COPY --from=builder /opt/ferrite/mistral_inference.component.wasm /opt/ferrite/mistral_inference.component.wasm

ENV FERRITE_MODEL_CACHE=/models

ENTRYPOINT ["/usr/local/bin/ferrite-rt"]
CMD ["run", "/opt/ferrite/mistral_inference.component.wasm"]
