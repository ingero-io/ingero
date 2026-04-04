# Ingero OSS — Multi-stage Docker Build with eBPF Compilation
#
# Compiles BPF probes from source using CO-RE for kernel portability.
# Build host must have BTF enabled (/sys/kernel/btf/vmlinux).
#
# Build:
#   docker build -t ingero-oss:latest .
#
# Run (observation mode):
#   docker run --privileged --pid=host \
#     -v /sys/kernel/btf:/sys/kernel/btf:ro \
#     -v /var/lib/ingero:/var/lib/ingero \
#     ingero-oss:latest trace --record
#
# Run (remediation mode — with orchestrator):
#   docker run --privileged --gpus all --pid=host \
#     -v /sys/kernel/btf:/sys/kernel/btf:ro \
#     -v /tmp:/tmp \
#     -v /dev/shm:/dev/shm \
#     -v /sys/fs/bpf:/sys/fs/bpf \
#     -v /var/lib/ingero:/var/lib/ingero \
#     ingero-oss:latest trace --record --remediate
#
# Note: --privileged is required for eBPF uprobe attachment to host libraries.
# On K8s with containerd, CAP_BPF+PERFMON+SYS_ADMIN may suffice (see EKS validation).

# ── Stage 1: Build ──────────────────────────────────────────────────
FROM golang:1.26-bookworm AS builder

# BPF compilation toolchain: clang-14 (BPF compiler), libbpf-dev (BPF headers),
# bpftool (vmlinux.h generation), llvm (BPF object tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang llvm libbpf-dev bpftool make pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install staticcheck for make test-ci
RUN go install honnef.co/go/tools/cmd/staticcheck@latest

WORKDIR /src

# Cache Go module downloads separately from source
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

# 1. Generate vmlinux.h from host kernel BTF (CO-RE portable)
# 2. Compile BPF C sources into Go-embedded objects via bpf2go
# 3. Build the Go binary with version injection
RUN make vmlinux \
    && make generate \
    && VERSION=${VERSION} COMMIT=${COMMIT} DATE=${BUILD_DATE} make build

# Run full test suite inside the build stage
RUN go test -v -race -count=1 -tags linux ./... \
    && go vet -tags linux ./... \
    && staticcheck ./...

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM alpine:3.20

# cilium/ebpf is pure Go — libbpf not needed at runtime.
# CGO_ENABLED=0 binary only needs libc6-compat for /proc parsing.
RUN apk add --no-cache ca-certificates libc6-compat

COPY --from=builder /src/bin/ingero /usr/local/bin/ingero

# NVIDIA Container Toolkit: inject GPU driver libs and utilities
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute

# Default DB path — mount a host volume to persist across restarts
ENV INGERO_DB=/var/lib/ingero/ingero.db
VOLUME /var/lib/ingero

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD ["/usr/local/bin/ingero", "version"]

LABEL org.opencontainers.image.title="Ingero" \
      org.opencontainers.image.description="Production GPU Causal Observability" \
      org.opencontainers.image.source="https://github.com/ingero-io/ingero"

ENTRYPOINT ["/usr/local/bin/ingero"]
CMD ["trace", "--record"]
