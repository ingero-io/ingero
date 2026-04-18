# Ingero v0.8 — Production GPU Causal Observability
# Multi-stage build: build Go binary, then minimal runtime image (~10MB)
#
# Required runtime flags for eBPF tracing:
#   docker run --privileged --pid=host \
#     -v /sys/kernel/debug:/sys/kernel/debug \
#     -v /sys/kernel/btf:/sys/kernel/btf:ro \
#     -v /var/lib/ingero:/var/lib/ingero \
#     ghcr.io/ingero-io/ingero:v0.8 trace --record
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
#     -v /sys/kernel/debug:/sys/kernel/debug \
#     -v /sys/kernel/btf:/sys/kernel/btf:ro \
#     ghcr.io/ingero-io/ingero trace                            # live tracing

FROM golang:1.26.2-bookworm AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
RUN CGO_ENABLED=0 go build -tags linux -trimpath \
    -ldflags="-s -w \
      -X github.com/ingero-io/ingero/internal/version.version=${VERSION} \
      -X github.com/ingero-io/ingero/internal/version.commit=${COMMIT} \
      -X github.com/ingero-io/ingero/internal/version.date=${BUILD_DATE}" \
    -o /ingero ./cmd/ingero/

FROM alpine:3.20
RUN apk add --no-cache ca-certificates libc6-compat
COPY --from=builder /ingero /usr/local/bin/ingero

# Tell NVIDIA Container Toolkit to inject GPU driver libs and utilities.
# Without these, nvidia-smi and libcuda.so are not mounted into the container.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute

# Default DB path inside the container — mount a host volume to persist data.
# Override with --db flag or INGERO_DB env var for multiple databases.
ENV INGERO_DB=/var/lib/ingero/ingero.db
VOLUME /var/lib/ingero

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD ["/usr/local/bin/ingero", "version"]

LABEL org.opencontainers.image.title="Ingero" \
      org.opencontainers.image.description="Production GPU Causal Observability" \
      org.opencontainers.image.source="https://github.com/ingero-io/ingero" \
      org.opencontainers.image.version="0.8.2" \
      io.modelcontextprotocol.server.name="io.github.ingero-io/ingero"

ENTRYPOINT ["/usr/local/bin/ingero"]
CMD ["trace", "--record"]
