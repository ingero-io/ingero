# Ingero v0.6 — Production GPU Causal Observability
# Multi-stage build: build Go binary, then minimal runtime image (~50MB)
#
# Usage:
#   docker run ingero/ingero demo --no-gpu                     # synthetic demo
#   docker run --privileged --pid=host ingero/ingero check     # system readiness check
#   docker run --privileged --pid=host ingero/ingero trace      # live tracing

FROM golang:1.26-bookworm AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN make generate build

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libbpf0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/bin/ingero /usr/local/bin/ingero

LABEL org.opencontainers.image.title="Ingero" \
      org.opencontainers.image.description="Production GPU Causal Observability" \
      org.opencontainers.image.source="https://github.com/ingero-io/ingero" \
      org.opencontainers.image.version="0.6.0"

ENTRYPOINT ["ingero"]
CMD ["demo", "--no-gpu"]
