#!/bin/bash
################################################################################
# Ingero TensorDock GPU VM Lifecycle Manager
#
# Automates TensorDock GPU VM creation, start/stop, and teardown for
# eBPF development sessions. Saves costs by stopping VMs when idle.
#
# Prerequisites:
#   - curl and jq installed
#   - TENSORDOCK_API_TOKEN set (env var or .env file)
#   - SSH key at ~/.ssh/id_ed25519.pub
#
# Usage:
#   ./scripts/tensordock/vm.sh deploy    # Create and provision a new GPU VM
#   ./scripts/tensordock/vm.sh start     # Start a stopped VM
#   ./scripts/tensordock/vm.sh stop      # Stop VM (reduces billing, preserves data)
#   ./scripts/tensordock/vm.sh destroy   # Delete VM permanently (fully stops billing)
#   ./scripts/tensordock/vm.sh destroy <id>  # Destroy by instance ID (for orphaned VMs)
#   ./scripts/tensordock/vm.sh status    # Show VM status
#   ./scripts/tensordock/vm.sh ssh       # SSH into the VM
#   ./scripts/tensordock/vm.sh info      # Show connection details and cost info
#
# Flags:
#   --force, -y    Skip confirmation prompts (for deploy, stop, destroy)
#
# Workflow:
#   deploy -> use -> destroy when done (VMs are ephemeral)
#
# State: VM details stored in .tensordock-vm.json (gitignored)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# ============================================================================
# Configuration
# ============================================================================

API_BASE="https://dashboard.tensordock.com/api/v2"

# VM defaults
VM_NAME="ingero-gpu-dev"
VM_VCPU=4
VM_RAM_GB=16
VM_STORAGE_GB=100
GPU_COUNT=1
PREFERRED_LOCATION_CITY="Manassas"

# GPU preference order: RTX 4090 (reliable), RTX 3090 (cheap fallback), H100 (scarce/flaky on TensorDock)
GPU_MODELS=("geforcertx4090-pcie-24gb" "geforcertx3090-pcie-24gb" "h100-sxm5-80gb")

# Image tiers: strongly prefer 22.04, only fall back to 24.04 with user confirmation
IMAGES_PREFERRED=("ubuntu2204_nvidia_550" "ubuntu2204")
IMAGES_FALLBACK=("ubuntu2404_ml_pytorch" "ubuntu2404_ml_everything" "ubuntu2404")

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to repo root (2 levels up from scripts/tensordock/).
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATE_FILE="$PROJECT_DIR/.tensordock-vm.json"
ENV_FILE="$PROJECT_DIR/.env"

# SSH
SSH_KEY_PATH="$HOME/.ssh/id_ed25519"
SSH_HOST_ALIAS="ingero-td"

# Timeouts
SSH_WAIT_TIMEOUT=300     # 5 minutes max to wait for SSH ready
SSH_WAIT_INTERVAL=10     # Poll every 10 seconds
API_POLL_INTERVAL=5      # Poll every 5 seconds for status changes
API_POLL_TIMEOUT=120     # 2 minutes max for status changes

# ============================================================================
# Utility Functions
# ============================================================================

load_token() {
    if [[ -n "$TENSORDOCK_API_TOKEN" ]]; then
        return 0
    fi

    if [[ -f "$ENV_FILE" ]]; then
        TENSORDOCK_API_TOKEN=$(grep -E '^TENSORDOCK_API_TOKEN=' "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        export TENSORDOCK_API_TOKEN
    fi

    if [[ -z "$TENSORDOCK_API_TOKEN" ]]; then
        print_error "TENSORDOCK_API_TOKEN not set."
        print_info "Set it via: export TENSORDOCK_API_TOKEN=your-token"
        print_info "Or add to .env: TENSORDOCK_API_TOKEN=your-token"
        print_info "Get your token at: https://dashboard.tensordock.com/developers"
        exit 1
    fi
}

check_prerequisites() {
    local missing=()

    if ! command -v curl &>/dev/null; then missing+=("curl"); fi
    if ! command -v jq &>/dev/null; then missing+=("jq"); fi
    if ! command -v ssh &>/dev/null; then missing+=("ssh"); fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing[*]}"
        print_info "Install via: sudo apt install ${missing[*]}"
        exit 1
    fi

    load_token

    local ssh_pub_key_path="${SSH_KEY_PATH}.pub"
    if [[ ! -f "$ssh_pub_key_path" ]]; then
        print_error "SSH public key not found: $ssh_pub_key_path"
        print_info "Generate one: ssh-keygen -t ed25519"
        exit 1
    fi
}

api_request() {
    local method="$1"
    local endpoint="$2"
    local body="${3:-}"

    local url="${API_BASE}${endpoint}"
    local curl_args=(
        -s
        -w "\n%{http_code}"
        -X "$method"
        -H "Authorization: Bearer $TENSORDOCK_API_TOKEN"
        -H "Content-Type: application/json"
        -H "Accept: application/json"
    )

    if [[ -n "$body" ]]; then
        curl_args+=(-d "$body")
    fi

    local response
    response=$(curl "${curl_args[@]}" "$url")

    local http_code
    http_code=$(echo "$response" | tail -1)
    local response_body
    response_body=$(echo "$response" | sed '$d')

    if [[ "$http_code" -ge 400 ]]; then
        print_error "API request failed (HTTP $http_code): $method $endpoint" >&2
        echo "$response_body" | jq '.' 2>/dev/null >&2 || echo "$response_body" >&2
        return 1
    fi

    echo "$response_body"
}

# ============================================================================
# State Management
# ============================================================================

save_state() {
    local instance_id="$1"
    local ip="$2"
    local ssh_port="$3"
    local status="$4"

    # Preserve original created timestamp, gpu_model, and dedicated_ip if state file already exists
    local created gpu_model dedicated_ip
    if [[ -f "$STATE_FILE" ]]; then
        created=$(jq -r '.created // empty' "$STATE_FILE" 2>/dev/null)
        gpu_model=$(jq -r '.gpu_model // empty' "$STATE_FILE" 2>/dev/null)
        dedicated_ip=$(jq -r '.dedicated_ip // empty' "$STATE_FILE" 2>/dev/null)
    fi
    created="${created:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
    gpu_model="${gpu_model:-unknown}"
    dedicated_ip="${dedicated_ip:-false}"

    jq -n \
        --arg id "$instance_id" \
        --arg ip "$ip" \
        --arg port "$ssh_port" \
        --arg status "$status" \
        --arg created "$created" \
        --arg gpu "$gpu_model" \
        --argjson dedicated_ip "$dedicated_ip" \
        '{instance_id: $id, ip: $ip, ssh_port: $port, status: $status, created: $created, gpu_model: $gpu, dedicated_ip: $dedicated_ip}' \
        > "$STATE_FILE"
}

load_state() {
    if [[ ! -f "$STATE_FILE" ]]; then
        return 1
    fi

    STATE_INSTANCE_ID=$(jq -r '.instance_id' "$STATE_FILE")
    STATE_IP=$(jq -r '.ip' "$STATE_FILE")
    STATE_SSH_PORT=$(jq -r '.ssh_port' "$STATE_FILE")
    STATE_STATUS=$(jq -r '.status' "$STATE_FILE")
    STATE_CREATED=$(jq -r '.created' "$STATE_FILE")
}

clear_state() {
    rm -f "$STATE_FILE"
}

# ============================================================================
# SSH Config Management
# ============================================================================

update_ssh_config() {
    local ip="$1"
    local port="$2"
    local ssh_config="$HOME/.ssh/config"

    mkdir -p "$HOME/.ssh"
    touch "$ssh_config"

    # Remove existing block if present
    if grep -q "# BEGIN ingero-td" "$ssh_config" 2>/dev/null; then
        local temp_file="${ssh_config}.tmp"
        sed '/# BEGIN ingero-td/,/# END ingero-td/d' "$ssh_config" > "$temp_file"
        mv "$temp_file" "$ssh_config"
    fi

    # Append managed block
    cat >> "$ssh_config" << EOF

# BEGIN ingero-td (managed by tensordock/vm.sh - do not edit manually)
Host ${SSH_HOST_ALIAS}
    HostName ${ip}
    Port ${port}
    User user
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
# END ingero-td
EOF

    print_success "SSH config updated: ssh ${SSH_HOST_ALIAS}"
}

remove_ssh_config() {
    local ssh_config="$HOME/.ssh/config"

    if [[ -f "$ssh_config" ]] && grep -q "# BEGIN ingero-td" "$ssh_config" 2>/dev/null; then
        local temp_file="${ssh_config}.tmp"
        sed '/# BEGIN ingero-td/,/# END ingero-td/d' "$ssh_config" > "$temp_file"
        mv "$temp_file" "$ssh_config"
        print_info "SSH config entry removed"
    fi
}

# ============================================================================
# SSH Wait
# ============================================================================

wait_for_ssh() {
    local ip="$1"
    local port="$2"
    local elapsed=0

    print_info "Waiting for VM to become SSH-ready (timeout: ${SSH_WAIT_TIMEOUT}s)..."

    while [[ $elapsed -lt $SSH_WAIT_TIMEOUT ]]; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -i "$SSH_KEY_PATH" -p "$port" "user@$ip" "echo ready" 2>/dev/null; then
            print_success "VM is SSH-ready!"
            return 0
        fi
        sleep "$SSH_WAIT_INTERVAL"
        elapsed=$((elapsed + SSH_WAIT_INTERVAL))
        print_info "  Still waiting... (${elapsed}s / ${SSH_WAIT_TIMEOUT}s)"
    done

    print_warn "VM did not become SSH-ready within ${SSH_WAIT_TIMEOUT}s"
    print_info "The VM may still be provisioning. Try: $0 ssh"
    return 1
}

# ============================================================================
# Location Discovery
# ============================================================================

find_location_ids() {
    # Returns newline-separated "location_id|gpu_model|network_type" tuples, ordered by preference:
    # For each GPU model (in preference order): preferred city, other US, global
    # network_type is "dedicated_ip" or "port_forward" based on location capabilities
    print_info "Querying TensorDock locations..." >&2

    local locations_response
    locations_response=$(api_request GET "/locations")

    local all_tuples=""
    local gpu_model

    for gpu_model in "${GPU_MODELS[@]}"; do
        local entries
        entries=$(echo "$locations_response" | jq -r \
            --arg city "$PREFERRED_LOCATION_CITY" \
            --arg gpu "$gpu_model" \
            '
            [.data.locations[] | select(.gpus[] | .v0Name == $gpu)] |
            ([ .[] | select(.city == $city) ] + [ .[] | select(.city != $city) | select(.country == "United States") ] + [ .[] | select(.country != "United States") ]) |
            .[] | .id + "|" + (if .network_features.port_forwarding_available then "port_forward" else "dedicated_ip" end)
            ' 2>/dev/null)

        local entry
        while IFS= read -r entry; do
            [[ -z "$entry" ]] && continue
            local loc_id net_type
            loc_id=$(echo "$entry" | cut -d'|' -f1)
            net_type=$(echo "$entry" | cut -d'|' -f2)
            local tuple="${loc_id}|${gpu_model}|${net_type}"
            if [[ -z "$all_tuples" ]]; then
                all_tuples="$tuple"
            else
                all_tuples="${all_tuples}"$'\n'"${tuple}"
            fi
        done <<< "$entries"
    done

    if [[ -z "$all_tuples" ]]; then
        print_error "No locations found with any supported GPU model" >&2
        print_info "Check availability at: https://dashboard.tensordock.com/deploy" >&2
        return 1
    fi

    local count
    count=$(echo "$all_tuples" | wc -l)
    print_success "Found ${count} location/GPU combination(s) to try" >&2

    echo "$all_tuples"
    return 0
}

# ============================================================================
# Extract connection details from instance response
# ============================================================================

extract_connection_details() {
    local detail_response="$1"

    # Try multiple response shapes (API may nest differently)
    EXTRACT_IP=$(echo "$detail_response" | jq -r '
        .data.attributes.ipAddress //
        .data.ipAddress //
        .ipAddress //
        empty' 2>/dev/null)

    EXTRACT_PORT=$(echo "$detail_response" | jq -r '
        [(.data.attributes.portForwards // .data.portForwards // .portForwards // [])[]
         | select(.internal_port == 22 or .internalPort == 22)
         | (.external_port // .externalPort)] | first // empty' 2>/dev/null)

    EXTRACT_STATUS=$(echo "$detail_response" | jq -r '
        .data.attributes.status //
        .data.status //
        .status //
        "unknown"' 2>/dev/null)

    # Fallback: if no port forwarding, assume direct SSH on port 22
    if [[ -z "$EXTRACT_PORT" || "$EXTRACT_PORT" == "null" ]]; then
        if [[ -n "$EXTRACT_IP" && "$EXTRACT_IP" != "null" ]]; then
            EXTRACT_PORT="22"
        fi
    fi
}

# ============================================================================
# Commands
# ============================================================================

cmd_deploy() {
    print_header "Deploying Ingero GPU VM"

    # Idempotency guard
    if load_state; then
        print_error "VM already exists: ${STATE_INSTANCE_ID}"
        print_info "Status: ${STATE_STATUS} | IP: ${STATE_IP}:${STATE_SSH_PORT}"
        print_info "Use '$0 destroy' first, or '$0 start' to resume."
        exit 1
    fi

    check_prerequisites

    # Read SSH public key
    local ssh_pub_key
    ssh_pub_key=$(cat "${SSH_KEY_PATH}.pub")

    # Find candidate location/GPU pairs (ordered by preference)
    local location_pairs
    location_pairs=$(find_location_ids) || exit 1

    # Confirm deployment
    print_warn "Cost Warning:"
    print_info "  GPUs (preference order): ${GPU_MODELS[*]}"
    print_info "  Resources: ${VM_VCPU} vCPU, ${VM_RAM_GB}GB RAM, ${VM_STORAGE_GB}GB SSD"
    print_info "  Billing starts immediately upon deployment."
    print_info "  Destroy when done: $0 destroy (fully stops billing)."
    echo ""
    if [[ "$FORCE_MODE" != "true" ]]; then
        read -p "Continue with deployment? [y/N] " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            print_info "Deployment cancelled."
            exit 0
        fi
    else
        print_info "Auto-confirmed (--force)"
    fi

    # Build cloud-init runcmd
    # NOTE: Does NOT clone the repo (private repo requires auth).
    # Instead, installs all dependencies so the VM is ready when you push code.
    # After deploy: scp/git push your code, then run scripts/validate-gpu-env.sh.
    local cloud_init_runcmd
    cloud_init_runcmd=$(jq -n '[
        "apt-mark hold grub-efi-amd64-signed 2>/dev/null || true",
        "apt-get update -qq",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang-14 llvm-14 libbpf-dev libelf-dev zlib1g-dev linux-tools-common linux-tools-generic make pkg-config git wget curl vim jq python3-pip build-essential stress-ng 2>&1 | tail -5",
        "apt-get install -y -qq linux-tools-$(uname -r) 2>/dev/null || true",
        "apt-get install -y -qq bpftrace trace-cmd 2>/dev/null || true",
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || (apt-get install -y nvidia-dkms-550 nvidia-driver-550 2>&1 | tail -5) || true",
        "mkdir -p /home/user/workspace",
        "case $(uname -m) in x86_64) GO_ARCH=amd64;; aarch64) GO_ARCH=arm64;; *) GO_ARCH=amd64;; esac && cd /tmp && wget -q https://go.dev/dl/go1.26.0.linux-${GO_ARCH}.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.26.0.linux-${GO_ARCH}.tar.gz && rm go1.26.0.linux-${GO_ARCH}.tar.gz",
        "grep -q /usr/local/go/bin /home/user/.bashrc || echo \"export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:\\$HOME/go/bin:\\$HOME/.local/bin:\\$PATH\" >> /home/user/.bashrc",
        "sudo -u user pip3 install --quiet torch torchvision numpy --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3 || true",
        "sudo -u user pip3 install --quiet transformers datasets diffusers accelerate 2>&1 | tail -3 || true",
        "apt-get install -y -qq libpython3.10-dbg 2>/dev/null || apt-get install -y -qq libpython3.12-dbg 2>/dev/null || true",
        "chown -R user:user /home/user/workspace",
        "echo \"Cloud-init complete: $(date)\" > /home/user/workspace/cloud-init-done.txt"
    ]')

    # Try deploying with image tiers: first all 22.04 images, then 24.04 only with confirmation
    local instance_id=""
    local deployed_gpu="" deployed_image="" deployed_location="" deployed_net_type=""

    # Helper: try a set of images across all location/GPU pairs
    # Returns 0 if VM created (sets deployed_* vars), 1 if all failed
    try_image_tier() {
        local -n images_ref=$1
        local tuple

        while IFS= read -r tuple; do
            [[ -z "$tuple" ]] && continue

            local location_id gpu_model net_type
            location_id=$(echo "$tuple" | cut -d'|' -f1)
            gpu_model=$(echo "$tuple" | cut -d'|' -f2)
            net_type=$(echo "$tuple" | cut -d'|' -f3)

            local image
            for image in "${images_ref[@]}"; do
                local request_body
                request_body=$(jq -n \
                    --arg name "$VM_NAME" \
                    --arg image "$image" \
                    --argjson vcpu "$VM_VCPU" \
                    --argjson ram "$VM_RAM_GB" \
                    --argjson storage "$VM_STORAGE_GB" \
                    --arg gpu_model "$gpu_model" \
                    --argjson gpu_count "$GPU_COUNT" \
                    --arg location_id "$location_id" \
                    --arg ssh_key "$ssh_pub_key" \
                    --argjson runcmd "$cloud_init_runcmd" \
                    --arg net_type "$net_type" \
                    '{
                        data: {
                            type: "virtualmachine",
                            attributes: {
                                type: "virtualmachine",
                                name: $name,
                                image: $image,
                                resources: {
                                    vcpu_count: $vcpu,
                                    ram_gb: $ram,
                                    storage_gb: $storage,
                                    gpus: {
                                        ($gpu_model): { count: $gpu_count }
                                    }
                                },
                                location_id: $location_id,
                                ssh_key: $ssh_key,
                                cloud_init: {
                                    packages: ["git", "curl", "jq", "vim"],
                                    package_update: true,
                                    runcmd: $runcmd
                                }
                            }
                        }
                    }')

                # Inject network config (jq can't conditionally add different shapes easily)
                if [[ "$net_type" == "dedicated_ip" ]]; then
                    request_body=$(echo "$request_body" | jq '.data.attributes.useDedicatedIp = true')
                else
                    request_body=$(echo "$request_body" | jq '.data.attributes.port_forwards = [{"internal_port": 22, "external_port": 20022, "protocol": "tcp"}]')
                fi

                print_info "Trying: ${gpu_model} in ${location_id} with ${image} (${net_type})..."
                local create_response
                create_response=$(api_request POST "/instances" "$request_body")

                instance_id=$(echo "$create_response" | jq -r '.data.id // .id // empty')

                if [[ -n "$instance_id" && "$instance_id" != "null" ]]; then
                    deployed_gpu="$gpu_model"
                    deployed_image="$image"
                    deployed_location="$location_id"
                    deployed_net_type="$net_type"
                    return 0
                fi

                # Check if error is image-related (try next image) vs node-related (try next location)
                local err_msg
                err_msg=$(echo "$create_response" | jq -r '.error // .message // "unknown error"' 2>/dev/null)
                if echo "$err_msg" | grep -qi "invalid_enum_value\|invalid.*image"; then
                    print_info "  Image ${image} not available here, trying next image..."
                    continue
                else
                    print_warn "  Failed: ${err_msg}. Trying next location..."
                    break  # skip remaining images for this location
                fi
            done
        done <<< "$location_pairs"

        return 1
    }

    # Pass 1: Try Ubuntu 22.04 images (preferred)
    print_info "Trying Ubuntu 22.04 images: ${IMAGES_PREFERRED[*]}"
    if ! try_image_tier IMAGES_PREFERRED; then
        # Pass 2: Ask user before falling back to Ubuntu 24.04
        print_warn "No Ubuntu 22.04 deployment available."
        print_warn "Ubuntu 24.04 images found: ${IMAGES_FALLBACK[*]}"
        print_info "Note: Ubuntu 24.04 may require extra setup (different package names, kernel version)."
        echo ""
        if [[ "$FORCE_MODE" != "true" ]]; then
            read -p "Try Ubuntu 24.04 instead? [y/N] " fallback_confirm
            if [[ "$fallback_confirm" != "y" && "$fallback_confirm" != "Y" ]]; then
                print_info "Deployment cancelled. Try again later when 22.04 is available."
                print_info "Check availability at: https://dashboard.tensordock.com/deploy"
                exit 0
            fi
        else
            print_info "Auto-confirmed 24.04 fallback (--force)"
        fi

        print_info "Trying Ubuntu 24.04 images: ${IMAGES_FALLBACK[*]}"
        if ! try_image_tier IMAGES_FALLBACK; then
            print_error "Failed to create VM in any available location."
            print_info "Check availability at: https://dashboard.tensordock.com/deploy"
            exit 1
        fi
    fi

    print_success "VM created: ${instance_id}"
    print_info "  GPU: ${deployed_gpu}, Image: ${deployed_image}, Location: ${deployed_location}"

    # Seed state file with gpu_model and dedicated_ip so save_state preserves them
    local is_dedicated_ip="false"
    if [[ "$deployed_net_type" == "dedicated_ip" ]]; then
        is_dedicated_ip="true"
    fi
    jq -n --arg gpu "$deployed_gpu" --argjson dip "$is_dedicated_ip" \
        '{gpu_model: $gpu, dedicated_ip: $dip}' > "$STATE_FILE"

    # Poll for running status and extract connection details
    print_info "Waiting for VM to start..."
    local ip="" ssh_port="" status=""
    local elapsed=0

    while [[ $elapsed -lt $API_POLL_TIMEOUT ]]; do
        sleep "$API_POLL_INTERVAL"
        elapsed=$((elapsed + API_POLL_INTERVAL))

        local detail_response
        detail_response=$(api_request GET "/instances/${instance_id}")

        extract_connection_details "$detail_response"
        ip="$EXTRACT_IP"
        ssh_port="$EXTRACT_PORT"
        status="$EXTRACT_STATUS"

        print_info "  Status: ${status} (${elapsed}s / ${API_POLL_TIMEOUT}s)"

        if [[ "$status" == "running" && -n "$ip" && "$ip" != "null" ]]; then
            break
        fi
    done

    if [[ -z "$ip" || "$ip" == "null" ]]; then
        print_error "Could not determine VM connection details."
        save_state "$instance_id" "unknown" "unknown" "$status"
        print_info "State saved. Check manually: $0 info"
        exit 1
    fi

    if [[ -z "$ssh_port" || "$ssh_port" == "null" ]]; then
        ssh_port="22"
    fi

    # Save state
    save_state "$instance_id" "$ip" "$ssh_port" "$status"
    print_success "State saved to ${STATE_FILE}"

    # Update SSH config
    update_ssh_config "$ip" "$ssh_port"

    # Wait for SSH
    wait_for_ssh "$ip" "$ssh_port" || true

    # Final output
    print_header "Deployment Complete"
    echo ""
    print_success "VM Details:"
    echo "  Instance ID:  ${instance_id}"
    echo "  IP:           ${ip}"
    echo "  SSH Port:     ${ssh_port}"
    echo "  Status:       ${status}"
    echo "  GPU:          ${deployed_gpu}"
    echo ""
    print_success "Connect:"
    echo "  ssh ${SSH_HOST_ALIAS}"
    echo "  # or: ssh -p ${ssh_port} user@${ip}"
    echo ""
    print_warn "Cost Reminder:"
    echo "  Billing is active while the VM is running."
    echo "  Destroy when done:  $0 destroy  (fully stops billing)"
    echo "  Stop (reduces billing, keeps data): $0 stop"
    echo ""
    print_info "Cloud-init is installing dependencies (Go, clang, PyTorch, libbpf)."
    print_info "Check if done: ssh ${SSH_HOST_ALIAS} 'cat ~/workspace/cloud-init-done.txt 2>/dev/null || echo still running'"
    print_info "After cloud-init: push your code, then run scripts/validate-gpu-env.sh"
}

cmd_start() {
    print_header "Starting Ingero GPU VM"

    if ! load_state; then
        print_error "No VM found. Deploy first: $0 deploy"
        exit 1
    fi

    load_token

    print_info "Starting VM: ${STATE_INSTANCE_ID}..."
    api_request POST "/instances/${STATE_INSTANCE_ID}/start" > /dev/null

    # Poll for running status
    local elapsed=0
    local status=""
    while [[ $elapsed -lt $API_POLL_TIMEOUT ]]; do
        sleep "$API_POLL_INTERVAL"
        elapsed=$((elapsed + API_POLL_INTERVAL))

        local detail_response
        detail_response=$(api_request GET "/instances/${STATE_INSTANCE_ID}")

        extract_connection_details "$detail_response"

        if [[ -n "$EXTRACT_IP" && "$EXTRACT_IP" != "null" ]]; then
            STATE_IP="$EXTRACT_IP"
        fi
        if [[ -n "$EXTRACT_PORT" && "$EXTRACT_PORT" != "null" ]]; then
            STATE_SSH_PORT="$EXTRACT_PORT"
        fi
        status="$EXTRACT_STATUS"

        print_info "  Status: ${status} (${elapsed}s)"

        if [[ "$status" == "running" ]]; then
            break
        fi
    done

    # Update state and SSH config (IP/port may change between stop/start)
    save_state "$STATE_INSTANCE_ID" "$STATE_IP" "$STATE_SSH_PORT" "$status"
    update_ssh_config "$STATE_IP" "$STATE_SSH_PORT"

    # Wait for SSH
    wait_for_ssh "$STATE_IP" "$STATE_SSH_PORT" || true

    print_success "VM started. Connect: ssh ${SSH_HOST_ALIAS}"
    print_warn "Billing is now active."
}

cmd_stop() {
    print_header "Stopping Ingero GPU VM"

    if ! load_state; then
        print_error "No VM found. Nothing to stop."
        exit 1
    fi

    load_token

    print_info "Stopping VM reduces billing (storage still billed). Data is preserved."
    print_info "To fully stop billing, use: $0 destroy"
    if [[ "$FORCE_MODE" != "true" ]]; then
        read -p "Stop VM ${STATE_INSTANCE_ID}? [y/N] " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            print_info "Cancelled."
            exit 0
        fi
    else
        print_info "Auto-confirmed (--force)"
    fi

    print_info "Stopping VM: ${STATE_INSTANCE_ID}..."
    api_request POST "/instances/${STATE_INSTANCE_ID}/stop" > /dev/null

    save_state "$STATE_INSTANCE_ID" "$STATE_IP" "$STATE_SSH_PORT" "stopped"

    print_success "VM stopped. Billing reduced (storage still billed)."
    print_info "Resume later: $0 start"
    print_info "Fully stop billing: $0 destroy"
}

cmd_destroy() {
    print_header "Destroying Ingero GPU VM"

    local target_id="${1:-}"

    if [[ -n "$target_id" ]]; then
        # Destroy by explicit instance ID (for orphaned VMs not in state file).
        load_token

        print_error "WARNING: This will permanently delete VM ${target_id} and all its data!"
        echo ""
        if [[ "$FORCE_MODE" != "true" ]]; then
            read -p "Type 'destroy' to confirm: " confirm
            if [[ "$confirm" != "destroy" ]]; then
                print_info "Cancelled."
                exit 0
            fi
        else
            print_info "Auto-confirmed (--force)"
        fi

        print_info "Deleting VM: ${target_id}..."
        api_request DELETE "/instances/${target_id}" > /dev/null

        # If the destroyed VM matches our state file, clean up state too.
        if load_state 2>/dev/null && [[ "${STATE_INSTANCE_ID}" == "${target_id}" ]]; then
            remove_ssh_config
            clear_state
            print_success "VM destroyed. State and SSH config cleaned up."
        else
            print_success "VM ${target_id} destroyed."
        fi
    else
        # Destroy the VM tracked in the state file.
        if ! load_state; then
            print_error "No VM found. Nothing to destroy."
            print_info "To destroy by ID: $0 destroy <instance-id>"
            exit 1
        fi

        load_token

        print_error "WARNING: This will permanently delete the VM and all its data!"
        print_info "Instance: ${STATE_INSTANCE_ID}"
        print_info "IP: ${STATE_IP}:${STATE_SSH_PORT}"
        echo ""
        if [[ "$FORCE_MODE" != "true" ]]; then
            read -p "Type 'destroy' to confirm: " confirm
            if [[ "$confirm" != "destroy" ]]; then
                print_info "Cancelled."
                exit 0
            fi
        else
            print_info "Auto-confirmed (--force)"
        fi

        print_info "Deleting VM: ${STATE_INSTANCE_ID}..."
        api_request DELETE "/instances/${STATE_INSTANCE_ID}" > /dev/null

        remove_ssh_config
        clear_state

        print_success "VM destroyed. State and SSH config cleaned up."
    fi
}

cmd_status() {
    if ! load_state; then
        print_info "No VM deployed. Deploy one: $0 deploy"
        exit 0
    fi

    load_token

    print_info "Fetching live status for ${STATE_INSTANCE_ID}..."

    local detail_response
    detail_response=$(api_request GET "/instances/${STATE_INSTANCE_ID}")

    extract_connection_details "$detail_response"

    # Update from live data if available
    if [[ -n "$EXTRACT_IP" && "$EXTRACT_IP" != "null" ]]; then
        STATE_IP="$EXTRACT_IP"
    fi
    if [[ -n "$EXTRACT_PORT" && "$EXTRACT_PORT" != "null" ]]; then
        STATE_SSH_PORT="$EXTRACT_PORT"
    fi

    # Update local state with live values
    save_state "$STATE_INSTANCE_ID" "$STATE_IP" "$STATE_SSH_PORT" "$EXTRACT_STATUS"

    print_header "Ingero GPU VM Status"
    echo "  Instance ID:  ${STATE_INSTANCE_ID}"
    echo "  Status:       ${EXTRACT_STATUS}"
    echo "  IP:           ${STATE_IP}"
    echo "  SSH Port:     ${STATE_SSH_PORT}"
    echo "  Created:      ${STATE_CREATED}"
    echo "  GPU:          $(jq -r '.gpu_model' "$STATE_FILE" 2>/dev/null || echo "$GPU_MODEL")"
    echo ""

    if [[ "$EXTRACT_STATUS" == "running" ]]; then
        print_success "VM is running. Connect: ssh ${SSH_HOST_ALIAS}"
        print_warn "Billing is active. Destroy when done: $0 destroy"
    elif [[ "$EXTRACT_STATUS" == "stopped" || "$EXTRACT_STATUS" == "Stopped" ]]; then
        print_info "VM is stopped. Billing reduced (storage still billed)."
        print_info "Resume: $0 start | Fully stop billing: $0 destroy"
    else
        print_warn "VM status: ${EXTRACT_STATUS}"
    fi
}

cmd_ssh() {
    if ! load_state; then
        print_error "No VM found. Deploy first: $0 deploy"
        exit 1
    fi

    exec ssh -p "$STATE_SSH_PORT" -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "user@$STATE_IP" "$@"
}

cmd_info() {
    if ! load_state; then
        print_info "No VM deployed."
        exit 0
    fi

    load_token

    # Fetch live details
    local detail_response
    detail_response=$(api_request GET "/instances/${STATE_INSTANCE_ID}" 2>/dev/null || echo "{}")

    local live_status rate_hourly
    live_status=$(echo "$detail_response" | jq -r '
        .data.attributes.status //
        .data.status //
        .status //
        "unknown"' 2>/dev/null)
    rate_hourly=$(echo "$detail_response" | jq -r '
        .data.attributes.rateHourly //
        .data.rateHourly //
        .rateHourly //
        "unknown"' 2>/dev/null)

    print_header "Ingero GPU VM Info"
    echo ""
    echo "  Instance ID:    ${STATE_INSTANCE_ID}"
    echo "  Status:         ${live_status}"
    echo "  IP Address:     ${STATE_IP}"
    echo "  SSH Port:       ${STATE_SSH_PORT}"
    echo "  Created:        ${STATE_CREATED}"
    echo "  GPU Model:      $(jq -r '.gpu_model' "$STATE_FILE" 2>/dev/null || echo "$GPU_MODEL")"
    echo "  Rate:           \$${rate_hourly}/hr"
    echo ""
    echo "  SSH Command:    ssh ${SSH_HOST_ALIAS}"
    echo "  Alt SSH:        ssh -p ${STATE_SSH_PORT} user@${STATE_IP}"
    echo "  VS Code:        code --remote ssh-remote+${SSH_HOST_ALIAS} /home/user/workspace/ingero"
    echo ""
    echo "  State File:     ${STATE_FILE}"
    echo ""

    if [[ "$live_status" == "running" ]]; then
        print_warn "Billing is active at \$${rate_hourly}/hr."
        print_info "Destroy when done: $0 destroy (fully stops billing)."
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    local command="${1:-help}"
    shift || true

    # Parse --force / -y flag from remaining args
    FORCE_MODE=false
    local args=()
    for arg in "$@"; do
        case "$arg" in
            --force|-y) FORCE_MODE=true ;;
            *) args+=("$arg") ;;
        esac
    done
    set -- "${args[@]}"

    case "$command" in
        deploy)   cmd_deploy "$@" ;;
        start)    cmd_start "$@" ;;
        stop)     cmd_stop "$@" ;;
        destroy)  cmd_destroy "$@" ;;
        status)   cmd_status "$@" ;;
        ssh)      cmd_ssh "$@" ;;
        info)     cmd_info "$@" ;;
        help|-h|--help)
            echo "Usage: $0 <command> [--force|-y]"
            echo ""
            echo "Commands:"
            echo "  deploy    Create and provision a new TensorDock GPU VM"
            echo "  start     Start a stopped VM (resumes billing)"
            echo "  stop      Stop a running VM (reduces billing, preserves data)"
            echo "  destroy [id]  Permanently delete the VM (fully stops billing)"
            echo "                If [id] given, destroys that instance (for orphaned VMs)"
            echo "  status    Show current VM status"
            echo "  ssh       Open SSH session to the VM"
            echo "  info      Show connection details and cost info"
            echo ""
            echo "Flags:"
            echo "  --force, -y  Skip confirmation prompts"
            echo ""
            echo "Workflow: deploy -> use -> destroy when done (VMs are ephemeral)"
            echo ""
            echo "Environment:"
            echo "  TENSORDOCK_API_TOKEN  API token (or set in .env file)"
            echo "  GPU models:           ${GPU_MODELS[*]}"
            echo "  Resources:            ${VM_VCPU} vCPU, ${VM_RAM_GB}GB RAM, ${VM_STORAGE_GB}GB SSD"
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Run '$0 help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
