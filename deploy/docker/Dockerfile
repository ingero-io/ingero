# Ingero v0.7 — Production GPU Causal Observability
# Multi-stage build: build Go binary, then minimal runtime image (~50MB)
#
# Required runtime flags for eBPF tracing:
#   docker run --privileged --pid=host \
#     -v /proc:/proc:ro \
#     -v /sys/kernel/debug:/sys/kernel/debug \
#     -v /sys/kernel/btf:/sys/kernel/btf:ro \
#     -v /var/lib/ingero:/var/lib/ingero \
#     ghcr.io/ingero-io/ingero:v0.7 trace --record
#
# Minimum capabilities (alternative to --privileged):
#   --cap-add=BPF --cap-add=PERFMON --cap-add=SYS_ADMIN
#
# Without --privileged or capabilities, only 'demo --no-gpu' and 'check' work.
#
# Usage examples:
#   docker run ghcr.io/ingero-io/ingero demo --no-gpu          # synthetic demo (no root needed)
#   docker run --privileged --pid=host ghcr.io/ingero-io/ingero check  # system readiness check
#   docker run --privileged --pid=host \
#     -v /proc:/proc:ro -v /sys/kernel/debug:/sys/kernel/debug \
#     -v /sys/kernel/btf:/sys/kernel/btf:ro \
#     ghcr.io/ingero-io/ingero trace                            # live tracing

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
      org.opencontainers.image.version="0.7.0"

ENTRYPOINT ["/usr/local/bin/ingero"]
CMD ["trace", "--record"]
