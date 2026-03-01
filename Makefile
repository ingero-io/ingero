.PHONY: all build generate clean test test-ci lint install uninstall wsl-setup gpu-deploy gpu-start gpu-stop gpu-destroy gpu-status gpu-ssh gpu-info gpu-validate gpu-logs lambda-deploy lambda-destroy lambda-status lambda-ssh lambda-info lambda-sync lambda-test lambda-validate lambda-logs azure-deploy azure-start azure-stop azure-destroy azure-status azure-ssh azure-info azure-sync azure-test azure-validate azure-logs fmt dev gpu-k3s-setup gpu-k3s-test lambda-k3s-setup lambda-k3s-test

# Variables
# Mono-repo root is one level above agent/.
REPO_ROOT := $(shell cd .. && pwd)
BINARY := bin/ingero
GOFLAGS := -tags linux
BPF_CLANG := clang
BPF_TARGET_ARCH := $(shell uname -m | sed -e 's/x86_64/x86/' -e 's/aarch64/arm64/')
BPF_CFLAGS := -O2 -g -target bpf -D__TARGET_ARCH_$(BPF_TARGET_ARCH)

# Version info injected at build time via Go linker flags
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE    ?= $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')

# Semantic version for README: 0.6.32 (base tag + commits-since-tag)
BASE_VER := $(shell echo $(VERSION) | sed 's/^v//; s/-.*//')
PATCH    := $(shell echo $(VERSION) | sed -n 's/^v[^-]*-\([0-9]*\)-.*/\1/p')
SEMVER   := $(BASE_VER).$(if $(PATCH),$(PATCH),0)

LDFLAGS := -ldflags "-X github.com/ingero-io/ingero/internal/version.version=$(VERSION) \
	-X github.com/ingero-io/ingero/internal/version.commit=$(COMMIT) \
	-X github.com/ingero-io/ingero/internal/version.date=$(DATE)"

# Single command to build everything
all: generate build test lint

# Generate eBPF Go bindings via bpf2go
# On ARM64 (GPU VMs like GH200), regenerate vmlinux.h first because the
# committed copy is from x86_64 WSL and rsync overwrites the cloud-init one.
generate:
ifeq ($(BPF_TARGET_ARCH),arm64)
	@echo "ARM64 detected — regenerating vmlinux.h from kernel BTF..."
	@$(MAKE) vmlinux
endif
	BPF_TARGET_ARCH=$(BPF_TARGET_ARCH) go generate ./internal/ebpf/...

# Build the agent binary (injects version from git at link time)
build:
	@mkdir -p bin
	go build $(GOFLAGS) $(LDFLAGS) -o $(BINARY) ./cmd/ingero/
	-@sed -i 's/^\*\*Version: [^*]*\*\*/**Version: $(SEMVER)**/' README.md

# Run tests
test:
	go test $(GOFLAGS) ./...

# CI target — comprehensive test suite
test-ci: generate build
	go test -v -race -count=1 $(GOFLAGS) ./...
	go vet $(GOFLAGS) ./...
	staticcheck ./...

# Run integration tests (requires root + CUDA)
test-integration:
	sudo go test $(GOFLAGS) -tags integration ./tests/integration/

# Lint
lint:
	staticcheck ./...

# Clean build artifacts
clean:
	rm -f $(BINARY)
	rm -f internal/ebpf/**/*_bpfel.go
	rm -f internal/ebpf/**/*_bpfel.o

# Install binary + systemd service
install: build
	sudo cp $(BINARY) /usr/local/bin/
	sudo mkdir -p /var/lib/ingero
	sudo cp deploy/systemd/ingero.service /etc/systemd/system/
	sudo systemctl daemon-reload
	@echo "Installed. Enable: sudo systemctl enable --now ingero"

# Uninstall binary + systemd service (preserves data at /var/lib/ingero/)
uninstall:
	-sudo systemctl stop ingero 2>/dev/null
	-sudo systemctl disable ingero 2>/dev/null
	sudo rm -f /etc/systemd/system/ingero.service
	sudo rm -f /usr/local/bin/ingero
	sudo systemctl daemon-reload
	@echo "Uninstalled. Data preserved at /var/lib/ingero/"

# Generate vmlinux.h from running kernel BTF
# Prefer bpftool matching the running kernel (avoids BTF version mismatch on GPU VMs
# where linux-tools-generic installs an old bpftool). Fall back to /usr/sbin/bpftool
# (system package), then any linux-tools version (WSL workaround).
vmlinux:
	@mkdir -p bpf/headers
	@BPFTOOL=$$(ls /usr/lib/linux-tools/$$(uname -r)/bpftool 2>/dev/null); \
	if [ -z "$$BPFTOOL" ]; then BPFTOOL=$$(which bpftool 2>/dev/null); fi; \
	if [ -z "$$BPFTOOL" ]; then BPFTOOL=$$(ls /usr/lib/linux-tools/*/bpftool 2>/dev/null | head -1); fi; \
	if [ -z "$$BPFTOOL" ]; then echo "ERROR: bpftool not found. Install: sudo apt install linux-tools-generic"; exit 1; fi; \
	echo "Using: $$BPFTOOL"; \
	$$BPFTOOL btf dump file /sys/kernel/btf/vmlinux format c > bpf/headers/vmlinux.h; \
	echo "Generated bpf/headers/vmlinux.h ($$(wc -l < bpf/headers/vmlinux.h) lines)"

# Format
fmt:
	gofumpt -w .
	find bpf/ -name '*.c' -o -name '*.h' | xargs clang-format -i

# Quick dev cycle: generate + build + run with sudo
dev: generate build
	sudo ./$(BINARY) check

# WSL development environment setup
wsl-setup:
	bash scripts/setup-wsl.sh

# TensorDock GPU VM lifecycle (--force skips confirmation prompts)
gpu-deploy:
	bash scripts/tensordock/vm.sh deploy --force

gpu-start:
	bash scripts/tensordock/vm.sh start --force

gpu-stop:
	bash scripts/tensordock/vm.sh stop --force

gpu-destroy:
	bash scripts/tensordock/vm.sh destroy --force

gpu-status:
	bash scripts/tensordock/vm.sh status

gpu-ssh:
	bash scripts/tensordock/vm.sh ssh

gpu-info:
	bash scripts/tensordock/vm.sh info

# Validate GPU VM environment (run on remote)
gpu-validate:
	bash scripts/tensordock/vm.sh ssh 'cd ~/workspace/ingero && bash scripts/validate-gpu-env.sh'

# Sync code to GPU VM and run full integration tests (one-shot)
# Runs make generate && make build first to regenerate eBPF objects for the VM's kernel
gpu-test: gpu-sync
	bash scripts/tensordock/vm.sh ssh 'export PATH=/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$$HOME/go/bin:$$HOME/.local/bin && cd ~/workspace/ingero && make generate && make build && bash scripts/gpu-test.sh'

# Transfer logs from TensorDock GPU VM to local logs/<datetime_provider_gpu>/ directory
gpu-logs:
	@GPU=$$(jq -r '.gpu_model // "unknown"' $(REPO_ROOT)/.tensordock-vm.json 2>/dev/null | sed 's/geforce//;s/-pcie.*//;s/-sxm.*//'); \
	DIR="logs/$$(date +%Y-%m-%d_%H-%M-%S)_tensordock_$$GPU"; \
	mkdir -p "$$DIR"; \
	IP=$$(jq -r .ip $(REPO_ROOT)/.tensordock-vm.json 2>/dev/null); \
	PORT=$$(jq -r .ssh_port $(REPO_ROOT)/.tensordock-vm.json 2>/dev/null); \
	if [ -z "$$IP" ] || [ "$$IP" = "null" ]; then echo "No VM deployed."; exit 1; fi; \
	echo "Transferring logs to $$DIR..."; \
	scp -P $$PORT -o StrictHostKeyChecking=no \
		"user@$$IP:~/workspace/ingero/logs/*.log" \
		"user@$$IP:~/workspace/ingero/logs/*.json" \
		"user@$$IP:~/workspace/ingero/logs/*.txt" \
		"user@$$IP:~/workspace/ingero/logs/*.out" \
		"user@$$IP:~/workspace/ingero/logs/*.db" \
		"user@$$IP:~/workspace/ingero/logs/*.md" \
		"$$DIR/" 2>/dev/null || true; \
	echo "Transferring SQLite DB..."; \
	ssh -p $$PORT -o StrictHostKeyChecking=no user@$$IP \
		'cp ~/.ingero/ingero.db /tmp/ingero.db 2>/dev/null || sudo cp /root/.ingero/ingero.db /tmp/ingero.db 2>/dev/null; chmod 644 /tmp/ingero.db 2>/dev/null' || true; \
	scp -P $$PORT -o StrictHostKeyChecking=no \
		user@$$IP:/tmp/ingero.db \
		"$$DIR/" 2>/dev/null || true; \
	echo "Logs transferred to $$DIR"; \
	ls -la "$$DIR/"

# Lambda Labs GPU VM lifecycle (H100 — no start/stop, only deploy/destroy)
lambda-deploy:
	bash scripts/lambdalabs/vm.sh deploy --force

lambda-destroy:
	bash scripts/lambdalabs/vm.sh destroy --force

lambda-status:
	bash scripts/lambdalabs/vm.sh status

lambda-ssh:
	bash scripts/lambdalabs/vm.sh ssh

lambda-info:
	bash scripts/lambdalabs/vm.sh info

# Validate Lambda Labs GPU VM environment (run on remote)
lambda-validate:
	bash scripts/lambdalabs/vm.sh ssh 'cd ~/workspace/ingero && bash scripts/validate-gpu-env.sh'

# Sync code to Lambda Labs VM and run full integration tests (one-shot)
# Runs make generate && make build first to regenerate eBPF objects for the VM's kernel
lambda-test: lambda-sync
	bash scripts/lambdalabs/vm.sh ssh 'export PATH=/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$$HOME/go/bin:$$HOME/.local/bin && cd ~/workspace/ingero && make generate && make build && bash scripts/gpu-test.sh'

# Transfer logs from Lambda Labs VM to local logs/<datetime_provider_gpu>/ directory
lambda-logs:
	@GPU=$$(jq -r '.instance_type // "unknown"' $(REPO_ROOT)/.lambdalabs-vm.json 2>/dev/null | sed 's/gpu_1x_//'); \
	DIR="logs/$$(date +%Y-%m-%d_%H-%M-%S)_lambda_$$GPU"; \
	mkdir -p "$$DIR"; \
	IP=$$(jq -r .ip $(REPO_ROOT)/.lambdalabs-vm.json 2>/dev/null); \
	if [ -z "$$IP" ] || [ "$$IP" = "null" ]; then echo "No Lambda instance deployed."; exit 1; fi; \
	echo "Transferring logs to $$DIR..."; \
	scp -o StrictHostKeyChecking=no \
		"ubuntu@$$IP:~/workspace/ingero/logs/*.log" \
		"ubuntu@$$IP:~/workspace/ingero/logs/*.json" \
		"ubuntu@$$IP:~/workspace/ingero/logs/*.txt" \
		"ubuntu@$$IP:~/workspace/ingero/logs/*.out" \
		"ubuntu@$$IP:~/workspace/ingero/logs/*.db" \
		"ubuntu@$$IP:~/workspace/ingero/logs/*.md" \
		"$$DIR/" 2>/dev/null || true; \
	echo "Transferring SQLite DB..."; \
	ssh -o StrictHostKeyChecking=no ubuntu@$$IP \
		'cp ~/.ingero/ingero.db /tmp/ingero.db 2>/dev/null || sudo cp /root/.ingero/ingero.db /tmp/ingero.db 2>/dev/null; chmod 644 /tmp/ingero.db 2>/dev/null' || true; \
	scp -o StrictHostKeyChecking=no \
		ubuntu@$$IP:/tmp/ingero.db \
		"$$DIR/" 2>/dev/null || true; \
	echo "Logs transferred to $$DIR"; \
	ls -la "$$DIR/"

# Sync code to Lambda Labs VM via rsync (port 22 direct, user ubuntu)
# Note: *_bpfel.go and *_bpfel.o are committed and required for build — do NOT exclude them
# Excludes logs/ to avoid overwriting VM test output with local log files
lambda-sync:
	@IP=$$(jq -r .ip $(REPO_ROOT)/.lambdalabs-vm.json 2>/dev/null); \
	if [ -z "$$IP" ] || [ "$$IP" = "null" ]; then echo "No Lambda instance deployed. Run: make lambda-deploy"; exit 1; fi; \
	echo "Syncing code to $$IP..."; \
	rsync -az --delete --exclude='.git' --exclude='/bin/' --exclude='/logs/' \
		-e "ssh -o StrictHostKeyChecking=no" \
		. ubuntu@$$IP:~/workspace/ingero/; \
	echo "Sync complete."

# Azure GPU VM lifecycle (deploy/stop/start/destroy + deallocate support)
azure-deploy:
	bash scripts/azure/vm.sh deploy --force

azure-start:
	bash scripts/azure/vm.sh start

azure-stop:
	bash scripts/azure/vm.sh stop --force

azure-destroy:
	bash scripts/azure/vm.sh destroy --force

azure-status:
	bash scripts/azure/vm.sh status

azure-ssh:
	bash scripts/azure/vm.sh ssh

azure-info:
	bash scripts/azure/vm.sh info

# Validate Azure GPU VM environment (run on remote)
azure-validate:
	bash scripts/azure/vm.sh ssh 'cd ~/workspace/ingero && bash scripts/validate-gpu-env.sh'

# Sync code to Azure VM and run full integration tests (one-shot)
azure-test: azure-sync
	bash scripts/azure/vm.sh ssh 'export PATH=/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$$HOME/go/bin:$$HOME/.local/bin && cd ~/workspace/ingero && make generate && make build && bash scripts/gpu-test.sh'

# Transfer logs from Azure VM to local logs/<datetime_provider_gpu>/ directory
azure-logs:
	@GPU=$$(jq -r '.vm_size // "unknown"' $(REPO_ROOT)/.azure-vm.json 2>/dev/null | sed 's/Standard_//;s/_v[0-9]*$$//'); \
	DIR="logs/$$(date +%Y-%m-%d_%H-%M-%S)_azure_$$GPU"; \
	mkdir -p "$$DIR"; \
	IP=$$(jq -r .ip $(REPO_ROOT)/.azure-vm.json 2>/dev/null); \
	if [ -z "$$IP" ] || [ "$$IP" = "null" ]; then echo "No Azure VM deployed."; exit 1; fi; \
	echo "Transferring logs to $$DIR..."; \
	scp -o StrictHostKeyChecking=no \
		"azureuser@$$IP:~/workspace/ingero/logs/*.log" \
		"azureuser@$$IP:~/workspace/ingero/logs/*.json" \
		"azureuser@$$IP:~/workspace/ingero/logs/*.txt" \
		"azureuser@$$IP:~/workspace/ingero/logs/*.out" \
		"azureuser@$$IP:~/workspace/ingero/logs/*.db" \
		"azureuser@$$IP:~/workspace/ingero/logs/*.md" \
		"$$DIR/" 2>/dev/null || true; \
	echo "Transferring SQLite DB..."; \
	ssh -o StrictHostKeyChecking=no azureuser@$$IP \
		'cp ~/.ingero/ingero.db /tmp/ingero.db 2>/dev/null || sudo cp /root/.ingero/ingero.db /tmp/ingero.db 2>/dev/null; chmod 644 /tmp/ingero.db 2>/dev/null' || true; \
	scp -o StrictHostKeyChecking=no \
		azureuser@$$IP:/tmp/ingero.db \
		"$$DIR/" 2>/dev/null || true; \
	echo "Logs transferred to $$DIR"; \
	ls -la "$$DIR/"

# Sync code to Azure VM via rsync (port 22 direct, user azureuser)
# Note: *_bpfel.go and *_bpfel.o are committed and required for build — do NOT exclude them
# Excludes logs/ to avoid overwriting VM test output with local log files
azure-sync:
	@IP=$$(jq -r .ip $(REPO_ROOT)/.azure-vm.json 2>/dev/null); \
	if [ -z "$$IP" ] || [ "$$IP" = "null" ]; then echo "No Azure VM deployed. Run: make azure-deploy"; exit 1; fi; \
	echo "Syncing code to $$IP..."; \
	rsync -az --delete --exclude='.git' --exclude='/bin/' --exclude='/logs/' \
		-e "ssh -o StrictHostKeyChecking=no" \
		. azureuser@$$IP:~/workspace/ingero/; \
	echo "Sync complete."

# Sync code to GPU VM via rsync
# Note: *_bpfel.go and *_bpfel.o are committed and required for build — do NOT exclude them
# Excludes logs/ to avoid overwriting VM test output with local log files
gpu-sync:
	@IP=$$(jq -r .ip $(REPO_ROOT)/.tensordock-vm.json 2>/dev/null); \
	PORT=$$(jq -r .ssh_port $(REPO_ROOT)/.tensordock-vm.json 2>/dev/null); \
	if [ -z "$$IP" ] || [ "$$IP" = "null" ]; then echo "No VM deployed. Run: make gpu-deploy"; exit 1; fi; \
	echo "Syncing code to $$IP:$$PORT..."; \
	rsync -az --delete --exclude='.git' --exclude='/bin/' --exclude='/logs/' \
		-e "ssh -p $$PORT -o StrictHostKeyChecking=no" \
		. user@$$IP:~/workspace/ingero/; \
	echo "Sync complete."

# k3s setup + test targets (run on GPU VM after sync)
# Install k3s + NVIDIA container toolkit + GPU device plugin
gpu-k3s-setup: gpu-sync
	bash scripts/tensordock/vm.sh ssh 'export PATH=/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$$HOME/go/bin:$$HOME/.local/bin && cd ~/workspace/ingero && bash scripts/k3s-setup.sh'

# Run k3s integration tests (deploy DaemonSet, run workload, verify cgroup_id)
gpu-k3s-test: gpu-sync
	bash scripts/tensordock/vm.sh ssh 'export PATH=/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$$HOME/go/bin:$$HOME/.local/bin && cd ~/workspace/ingero && bash scripts/k3s-test.sh'

# Lambda Labs k3s targets
lambda-k3s-setup: lambda-sync
	bash scripts/lambdalabs/vm.sh ssh 'export PATH=/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$$HOME/go/bin:$$HOME/.local/bin && cd ~/workspace/ingero && bash scripts/k3s-setup.sh'

lambda-k3s-test: lambda-sync
	bash scripts/lambdalabs/vm.sh ssh 'export PATH=/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$$HOME/go/bin:$$HOME/.local/bin && cd ~/workspace/ingero && bash scripts/k3s-test.sh'
