#!/bin/bash
################################################################################
# Ingero Lambda Labs GPU VM Lifecycle Manager
#
# Automates Lambda Labs H100 instance creation and teardown for GPU testing.
# Lambda Labs does NOT support stop/resume — instances can only be launched
# or terminated. Billing runs from launch to terminate.
#
# Prerequisites:
#   - curl and jq installed
#   - LAMBDALABS_API_TOKEN set (env var or .env file)
#   - SSH key at ~/.ssh/id_ed25519.pub
#
# Usage:
#   ./scripts/lambdalabs/vm.sh deploy    # Launch an H100 instance
#   ./scripts/lambdalabs/vm.sh destroy   # Terminate instance (stops billing)
#   ./scripts/lambdalabs/vm.sh status    # Show instance status
#   ./scripts/lambdalabs/vm.sh ssh       # SSH into the instance
#   ./scripts/lambdalabs/vm.sh info      # Show connection details and cost
#
# Flags:
#   --force, -y    Skip confirmation prompts
#
# State: Instance details stored in .lambdalabs-vm.json (gitignored)
#
# API: Lambda Cloud API v1.9.3
#   Docs: https://docs-api.lambda.ai/api/cloud
#   Base: https://cloud.lambdalabs.com/api/v1
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# ============================================================================
# Configuration
# ============================================================================

API_BASE="https://cloud.lambdalabs.com/api/v1"

# Instance defaults
INSTANCE_NAME="ingero-gpu-dev"
SSH_KEY_NAME="ingero-dev"

# Instance type preference order (override with LAMBDA_INSTANCE_TYPES env var)
# Default: H100 > A100 > A10 > L40 > L4 — broadest-to-cheapest fallback chain.
# Any GPU with 24+ GB VRAM, CUDA support, and a working NVIDIA driver works for Ingero.
if [[ -n "${LAMBDA_INSTANCE_TYPES:-}" ]]; then
    IFS=',' read -ra INSTANCE_TYPES <<< "$LAMBDA_INSTANCE_TYPES"
else
    INSTANCE_TYPES=(
        "gpu_1x_h100_sxm5"
        "gpu_1x_h100_pcie"
        "gpu_1x_a100_sxm4"
        "gpu_1x_a100"
        "gpu_1x_a10"
        "gpu_1x_l40"
        "gpu_1x_l4"
    )
fi

# Region preference: US East first, then other US regions, then any
PREFERRED_REGIONS=("us-east-1" "us-west-1" "us-east-2" "us-south-1" "us-midwest-1" "us-south-2" "us-south-3" "us-west-2" "us-west-3")

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to mono-repo root (3 levels up from agent/scripts/lambdalabs/).
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STATE_FILE="$PROJECT_DIR/.lambdalabs-vm.json"
ENV_FILE="$PROJECT_DIR/.env"

# SSH
SSH_KEY_PATH="$HOME/.ssh/id_ed25519"
SSH_HOST_ALIAS="ingero-lambda"
SSH_USER="ubuntu"

# Timeouts
SSH_WAIT_TIMEOUT=300
SSH_WAIT_INTERVAL=10
API_POLL_INTERVAL=5
API_POLL_TIMEOUT=600    # 10 min — Lambda H100 boot can take 5-10 min (API lags behind actual state)

# ============================================================================
# Utility Functions
# ============================================================================

load_token() {
    if [[ -n "$LAMBDALABS_API_TOKEN" ]]; then
        return 0
    fi

    if [[ -f "$ENV_FILE" ]]; then
        LAMBDALABS_API_TOKEN=$(grep -E '^LAMBDALABS_API_TOKEN=' "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        export LAMBDALABS_API_TOKEN
    fi

    if [[ -z "$LAMBDALABS_API_TOKEN" ]]; then
        print_error "LAMBDALABS_API_TOKEN not set."
        print_info "Set it via: export LAMBDALABS_API_TOKEN=your-key"
        print_info "Or add to .env: LAMBDALABS_API_TOKEN=your-key"
        print_info "Get your key at: https://cloud.lambdalabs.com/api-keys"
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
        -u "$LAMBDALABS_API_TOKEN:"
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
    local status="$3"
    local instance_type="$4"
    local region="$5"
    local rate_cents="$6"

    local created
    if [[ -f "$STATE_FILE" ]]; then
        created=$(jq -r '.created // empty' "$STATE_FILE" 2>/dev/null)
    fi
    created="${created:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

    jq -n \
        --arg id "$instance_id" \
        --arg ip "$ip" \
        --arg status "$status" \
        --arg type "$instance_type" \
        --arg region "$region" \
        --argjson rate "${rate_cents:-0}" \
        --arg created "$created" \
        '{instance_id: $id, ip: $ip, status: $status, instance_type: $type, region: $region, rate_cents_per_hour: $rate, created: $created}' \
        > "$STATE_FILE"
}

load_state() {
    if [[ ! -f "$STATE_FILE" ]]; then
        return 1
    fi

    STATE_INSTANCE_ID=$(jq -r '.instance_id' "$STATE_FILE")
    STATE_IP=$(jq -r '.ip' "$STATE_FILE")
    STATE_STATUS=$(jq -r '.status' "$STATE_FILE")
    STATE_TYPE=$(jq -r '.instance_type' "$STATE_FILE")
    STATE_REGION=$(jq -r '.region' "$STATE_FILE")
    STATE_RATE=$(jq -r '.rate_cents_per_hour' "$STATE_FILE")
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
    local ssh_config="$HOME/.ssh/config"

    mkdir -p "$HOME/.ssh"
    touch "$ssh_config"

    # Remove existing block if present
    if grep -q "# BEGIN ingero-lambda" "$ssh_config" 2>/dev/null; then
        local temp_file="${ssh_config}.tmp"
        sed '/# BEGIN ingero-lambda/,/# END ingero-lambda/d' "$ssh_config" > "$temp_file"
        mv "$temp_file" "$ssh_config"
    fi

    # Append managed block — Lambda uses port 22, user 'ubuntu'
    cat >> "$ssh_config" << EOF

# BEGIN ingero-lambda (managed by lambdalabs/vm.sh - do not edit manually)
Host ${SSH_HOST_ALIAS}
    HostName ${ip}
    Port 22
    User ${SSH_USER}
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
# END ingero-lambda
EOF

    print_success "SSH config updated: ssh ${SSH_HOST_ALIAS}"
}

remove_ssh_config() {
    local ssh_config="$HOME/.ssh/config"

    if [[ -f "$ssh_config" ]] && grep -q "# BEGIN ingero-lambda" "$ssh_config" 2>/dev/null; then
        local temp_file="${ssh_config}.tmp"
        sed '/# BEGIN ingero-lambda/,/# END ingero-lambda/d' "$ssh_config" > "$temp_file"
        mv "$temp_file" "$ssh_config"
        print_info "SSH config entry removed"
    fi
}

# ============================================================================
# SSH Wait
# ============================================================================

wait_for_ssh() {
    local ip="$1"
    local elapsed=0

    print_info "Waiting for instance to become SSH-ready (timeout: ${SSH_WAIT_TIMEOUT}s)..."

    while [[ $elapsed -lt $SSH_WAIT_TIMEOUT ]]; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -i "$SSH_KEY_PATH" "${SSH_USER}@${ip}" "echo ready" 2>/dev/null; then
            print_success "Instance is SSH-ready!"
            return 0
        fi
        sleep "$SSH_WAIT_INTERVAL"
        elapsed=$((elapsed + SSH_WAIT_INTERVAL))
        print_info "  Still waiting... (${elapsed}s / ${SSH_WAIT_TIMEOUT}s)"
    done

    print_warn "Instance did not become SSH-ready within ${SSH_WAIT_TIMEOUT}s"
    print_info "The instance may still be booting. Try: $0 ssh"
    return 1
}

# ============================================================================
# SSH Key Management
# ============================================================================

ensure_ssh_key() {
    print_info "Checking SSH keys on Lambda account..."

    local keys_response
    keys_response=$(api_request GET "/ssh-keys")

    # Check if our key already exists by name
    local existing_key
    existing_key=$(echo "$keys_response" | jq -r --arg name "$SSH_KEY_NAME" \
        '.data // [] | map(select(.name == $name)) | .[0].name // empty' 2>&1) || true

    if [[ -n "$existing_key" ]]; then
        print_info "SSH key '${SSH_KEY_NAME}' already exists on Lambda account"
        return 0
    fi

    # Also check by public key content — prevents duplicate uploads even if
    # the name check fails or the key was renamed on the dashboard.
    local ssh_pub_key
    ssh_pub_key=$(cat "${SSH_KEY_PATH}.pub")
    local key_material
    key_material=$(echo "$ssh_pub_key" | awk '{print $2}')

    local existing_by_content
    existing_by_content=$(echo "$keys_response" | jq -r --arg km "$key_material" \
        '.data // [] | map(select(.public_key | contains($km))) | .[0].name // empty' 2>&1) || true

    if [[ -n "$existing_by_content" ]]; then
        print_info "SSH key already on Lambda account as '${existing_by_content}'"
        # Use the existing name for the launch request
        SSH_KEY_NAME="$existing_by_content"
        return 0
    fi

    # Safety check: if we couldn't parse the key list at all, don't blindly upload.
    local key_count
    key_count=$(echo "$keys_response" | jq '.data | length' 2>/dev/null || echo "-1")
    if [[ "$key_count" == "-1" ]]; then
        print_warn "Could not parse SSH key list from Lambda API. Assuming key '${SSH_KEY_NAME}' exists."
        print_warn "If launch fails with 'ssh key not found', add your key manually at:"
        print_warn "  https://cloud.lambdalabs.com/ssh-keys"
        return 0
    fi

    # Key truly does not exist — upload it
    print_info "Uploading SSH key '${SSH_KEY_NAME}' to Lambda account..."
    local add_body
    add_body=$(jq -n \
        --arg name "$SSH_KEY_NAME" \
        --arg key "$ssh_pub_key" \
        '{name: $name, public_key: $key}')

    api_request POST "/ssh-keys" "$add_body" > /dev/null
    print_success "SSH key '${SSH_KEY_NAME}' uploaded"
}

# ============================================================================
# Instance Type Discovery
# ============================================================================

find_available_instance() {
    print_info "Querying Lambda Labs for GPU availability (types: ${INSTANCE_TYPES[*]})..."

    local types_response
    types_response=$(api_request GET "/instance-types")

    # For each instance type in preference order, check for available regions
    for instance_type in "${INSTANCE_TYPES[@]}"; do
        local type_data
        type_data=$(echo "$types_response" | jq --arg t "$instance_type" '.data[$t] // empty' 2>/dev/null)

        if [[ -z "$type_data" || "$type_data" == "null" ]]; then
            print_info "  ${instance_type}: not listed"
            continue
        fi

        local available_regions
        available_regions=$(echo "$type_data" | jq -r '.regions_with_capacity_available[].name' 2>/dev/null)

        if [[ -z "$available_regions" ]]; then
            print_info "  ${instance_type}: sold out everywhere"
            continue
        fi

        # Try preferred regions first
        for region in "${PREFERRED_REGIONS[@]}"; do
            if echo "$available_regions" | grep -qx "$region"; then
                FOUND_TYPE="$instance_type"
                FOUND_REGION="$region"
                FOUND_PRICE=$(echo "$type_data" | jq '.instance_type.price_cents_per_hour' 2>/dev/null)
                FOUND_DESC=$(echo "$type_data" | jq -r '.instance_type.description' 2>/dev/null)
                FOUND_VCPUS=$(echo "$type_data" | jq '.instance_type.specs.vcpus' 2>/dev/null)
                FOUND_RAM=$(echo "$type_data" | jq '.instance_type.specs.memory_gib' 2>/dev/null)
                FOUND_STORAGE=$(echo "$type_data" | jq '.instance_type.specs.storage_gib' 2>/dev/null)
                print_success "Found: ${FOUND_DESC} in ${FOUND_REGION}"
                return 0
            fi
        done

        # Fall back to any available region
        local first_region
        first_region=$(echo "$available_regions" | head -1)
        FOUND_TYPE="$instance_type"
        FOUND_REGION="$first_region"
        FOUND_PRICE=$(echo "$type_data" | jq '.instance_type.price_cents_per_hour' 2>/dev/null)
        FOUND_DESC=$(echo "$type_data" | jq -r '.instance_type.description' 2>/dev/null)
        FOUND_VCPUS=$(echo "$type_data" | jq '.instance_type.specs.vcpus' 2>/dev/null)
        FOUND_RAM=$(echo "$type_data" | jq '.instance_type.specs.memory_gib' 2>/dev/null)
        FOUND_STORAGE=$(echo "$type_data" | jq '.instance_type.specs.storage_gib' 2>/dev/null)
        print_success "Found: ${FOUND_DESC} in ${FOUND_REGION}"
        return 0
    done

    print_error "No instances available in any region for types: ${INSTANCE_TYPES[*]}"
    print_info "Instance types checked: ${INSTANCE_TYPES[*]}"
    print_info "Check availability at: https://cloud.lambdalabs.com/instances"
    return 1
}

# ============================================================================
# Commands
# ============================================================================

cmd_deploy() {
    print_header "Deploying Ingero GPU Instance (Lambda Labs)"

    # Idempotency guard
    if load_state; then
        print_error "Instance already exists: ${STATE_INSTANCE_ID}"
        print_info "Status: ${STATE_STATUS} | IP: ${STATE_IP}"
        print_info "Use '$0 destroy' first."
        exit 1
    fi

    check_prerequisites
    ensure_ssh_key
    find_available_instance || exit 1

    # Format price
    local price_dollars
    price_dollars=$(awk "BEGIN {printf \"%.2f\", ${FOUND_PRICE}/100}")

    # Confirm
    print_warn "Cost Warning:"
    print_info "  GPU: ${FOUND_DESC}"
    print_info "  Region: ${FOUND_REGION}"
    print_info "  Specs: ${FOUND_VCPUS} vCPU, ${FOUND_RAM}GB RAM, ${FOUND_STORAGE}GB SSD"
    print_info "  Rate: \$${price_dollars}/hr"
    print_info "  NO pause/resume — billing runs until you destroy."
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

    # Build cloud-init user_data
    local user_data
    user_data=$(cat << 'CLOUDINIT'
#!/bin/bash
set -e
mkdir -p /home/ubuntu/workspace
chown ubuntu:ubuntu /home/ubuntu/workspace
exec > /home/ubuntu/workspace/setup.log 2>&1
echo "=== Cloud-init started: $(date) ==="

# System packages — everything an ML engineer needs to build+test Ingero
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential clang-14 llvm-14 libbpf-dev libelf-dev zlib1g-dev \
    linux-tools-common linux-tools-generic \
    make pkg-config git wget curl vim jq sqlite3 \
    python3-pip stress-ng 2>&1 | tail -5

apt-get install -y -qq linux-tools-$(uname -r) 2>/dev/null || true
apt-get install -y -qq bpftrace trace-cmd 2>/dev/null || true

# Go
GO_ARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') && \
    cd /tmp && wget -q https://go.dev/dl/go1.26.0.linux-${GO_ARCH}.tar.gz \
    && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.26.0.linux-${GO_ARCH}.tar.gz \
    && rm go1.26.0.linux-${GO_ARCH}.tar.gz

grep -q /usr/local/go/bin /home/ubuntu/.bashrc || \
    echo 'export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/go/bin:$HOME/.local/bin:$PATH' >> /home/ubuntu/.bashrc

# PyTorch (Lambda Stack may already have this)
sudo -u ubuntu pip3 install --quiet torch torchvision numpy --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3 || true
sudo -u ubuntu pip3 install --quiet transformers datasets diffusers accelerate 2>&1 | tail -3 || true

# Python debug symbols for DWARF-based CPython frame extraction
apt-get install -y -qq libpython3.10-dbg 2>/dev/null || apt-get install -y -qq libpython3.12-dbg 2>/dev/null || true

# Ensure workspace ownership (may have been created by root during package installs)
chown -R ubuntu:ubuntu /home/ubuntu/workspace

echo "=== Cloud-init complete: $(date) ==="
echo "Cloud-init complete: $(date)" > /home/ubuntu/workspace/cloud-init-done.txt
CLOUDINIT
)

    # Launch
    print_info "Launching ${FOUND_TYPE} in ${FOUND_REGION}..."

    local request_body
    request_body=$(jq -n \
        --arg region "$FOUND_REGION" \
        --arg type "$FOUND_TYPE" \
        --arg key "$SSH_KEY_NAME" \
        --arg name "$INSTANCE_NAME" \
        --arg user_data "$user_data" \
        '{
            region_name: $region,
            instance_type_name: $type,
            ssh_key_names: [$key],
            name: $name,
            user_data: $user_data
        }')

    local launch_response
    launch_response=$(api_request POST "/instance-operations/launch" "$request_body")

    local instance_id
    instance_id=$(echo "$launch_response" | jq -r '.data.instance_ids[0]')

    if [[ -z "$instance_id" || "$instance_id" == "null" ]]; then
        print_error "Failed to launch instance."
        echo "$launch_response" | jq '.' 2>/dev/null || echo "$launch_response"
        exit 1
    fi

    print_success "Instance launched: ${instance_id}"

    # Save initial state
    save_state "$instance_id" "" "booting" "$FOUND_TYPE" "$FOUND_REGION" "$FOUND_PRICE"

    # Poll for active status
    print_info "Waiting for instance to become active..."
    local ip="" status=""
    local elapsed=0

    while [[ $elapsed -lt $API_POLL_TIMEOUT ]]; do
        sleep "$API_POLL_INTERVAL"
        elapsed=$((elapsed + API_POLL_INTERVAL))

        local detail_response
        detail_response=$(api_request GET "/instances/${instance_id}")

        ip=$(echo "$detail_response" | jq -r '.data.ip // empty' 2>/dev/null)
        status=$(echo "$detail_response" | jq -r '.data.status // "unknown"' 2>/dev/null)

        print_info "  Status: ${status} (${elapsed}s / ${API_POLL_TIMEOUT}s)"

        if [[ "$status" == "active" && -n "$ip" && "$ip" != "null" ]]; then
            break
        fi

        if [[ "$status" == "terminated" || "$status" == "unhealthy" ]]; then
            print_error "Instance entered ${status} state."
            save_state "$instance_id" "${ip:-}" "$status" "$FOUND_TYPE" "$FOUND_REGION" "$FOUND_PRICE"
            exit 1
        fi
    done

    if [[ -z "$ip" || "$ip" == "null" ]]; then
        print_error "Could not determine instance IP within ${API_POLL_TIMEOUT}s."
        save_state "$instance_id" "" "$status" "$FOUND_TYPE" "$FOUND_REGION" "$FOUND_PRICE"
        print_info "State saved. Check: $0 status"
        exit 1
    fi

    # Save state with IP
    save_state "$instance_id" "$ip" "$status" "$FOUND_TYPE" "$FOUND_REGION" "$FOUND_PRICE"
    print_success "State saved to ${STATE_FILE}"

    # Update SSH config
    update_ssh_config "$ip"

    # Wait for SSH
    wait_for_ssh "$ip" || true

    # Final output
    print_header "Deployment Complete"
    echo ""
    print_success "Instance Details:"
    echo "  Instance ID:  ${instance_id}"
    echo "  IP:           ${ip}"
    echo "  Status:       ${status}"
    echo "  GPU:          ${FOUND_DESC}"
    echo "  Region:       ${FOUND_REGION}"
    echo "  Rate:         \$${price_dollars}/hr"
    echo ""
    print_success "Connect:"
    echo "  ssh ${SSH_HOST_ALIAS}"
    echo "  # or: ssh ${SSH_USER}@${ip}"
    echo ""
    print_warn "Cost Reminder:"
    echo "  Billing is active at \$${price_dollars}/hr."
    echo "  Lambda Labs has NO pause/resume — destroy when done."
    echo "  Destroy: $0 destroy"
    echo ""
    print_info "Cloud-init is installing dependencies (Go, clang, PyTorch, libbpf)."
    print_info "Check if done: ssh ${SSH_HOST_ALIAS} 'cat ~/workspace/cloud-init-done.txt 2>/dev/null || echo still running'"
    print_info "Next steps after cloud-init:"
    print_info "  1. Sync code:  make lambda-sync"
    print_info "  2. Validate:   make lambda-validate"
    print_info "  3. Or SSH in:  ssh ${SSH_HOST_ALIAS}"
}

cmd_destroy() {
    print_header "Destroying Ingero GPU Instance (Lambda Labs)"

    if ! load_state; then
        print_error "No instance found. Nothing to destroy."
        exit 1
    fi

    load_token

    print_error "WARNING: This will permanently terminate the instance and all its data!"
    print_info "Instance: ${STATE_INSTANCE_ID}"
    print_info "IP: ${STATE_IP}"
    print_info "Type: ${STATE_TYPE}"
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

    print_info "Terminating instance: ${STATE_INSTANCE_ID}..."

    local terminate_body
    terminate_body=$(jq -n --arg id "$STATE_INSTANCE_ID" '{"instance_ids": [$id]}')

    api_request POST "/instance-operations/terminate" "$terminate_body" > /dev/null

    remove_ssh_config
    clear_state

    print_success "Instance terminated. State and SSH config cleaned up."
    print_info "Billing has stopped."
}

cmd_status() {
    if ! load_state; then
        print_info "No instance deployed. Deploy one: $0 deploy"
        exit 0
    fi

    load_token

    print_info "Fetching live status for ${STATE_INSTANCE_ID}..."

    local detail_response
    detail_response=$(api_request GET "/instances/${STATE_INSTANCE_ID}")

    local live_status live_ip
    live_status=$(echo "$detail_response" | jq -r '.data.status // "unknown"' 2>/dev/null)
    live_ip=$(echo "$detail_response" | jq -r '.data.ip // empty' 2>/dev/null)

    # Update local state
    if [[ -n "$live_ip" && "$live_ip" != "null" ]]; then
        STATE_IP="$live_ip"
    fi
    save_state "$STATE_INSTANCE_ID" "$STATE_IP" "$live_status" "$STATE_TYPE" "$STATE_REGION" "$STATE_RATE"

    print_header "Ingero GPU Instance Status (Lambda Labs)"
    echo "  Instance ID:  ${STATE_INSTANCE_ID}"
    echo "  Status:       ${live_status}"
    echo "  IP:           ${STATE_IP}"
    echo "  Type:         ${STATE_TYPE}"
    echo "  Region:       ${STATE_REGION}"
    echo "  Created:      ${STATE_CREATED}"
    echo ""

    if [[ "$live_status" == "active" ]]; then
        print_success "Instance is running. Connect: ssh ${SSH_HOST_ALIAS}"
        local price_dollars
        price_dollars=$(awk "BEGIN {printf \"%.2f\", ${STATE_RATE}/100}")
        print_warn "Billing is active at \$${price_dollars}/hr. Destroy when done: $0 destroy"
    elif [[ "$live_status" == "booting" ]]; then
        print_info "Instance is still booting..."
    elif [[ "$live_status" == "terminated" ]]; then
        print_info "Instance has been terminated."
        print_info "Clean up state: rm ${STATE_FILE}"
    else
        print_warn "Instance status: ${live_status}"
    fi
}

cmd_ssh() {
    if ! load_state; then
        print_error "No instance found. Deploy first: $0 deploy"
        exit 1
    fi

    exec ssh -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${SSH_USER}@${STATE_IP}" "$@"
}

cmd_info() {
    if ! load_state; then
        print_info "No instance deployed."
        exit 0
    fi

    load_token

    # Fetch live details
    local detail_response
    detail_response=$(api_request GET "/instances/${STATE_INSTANCE_ID}" 2>/dev/null || echo "{}")

    local live_status
    live_status=$(echo "$detail_response" | jq -r '.data.status // "unknown"' 2>/dev/null)

    local price_dollars
    price_dollars=$(awk "BEGIN {printf \"%.2f\", ${STATE_RATE}/100}")

    print_header "Ingero GPU Instance Info (Lambda Labs)"
    echo ""
    echo "  Instance ID:    ${STATE_INSTANCE_ID}"
    echo "  Status:         ${live_status}"
    echo "  IP Address:     ${STATE_IP}"
    echo "  Instance Type:  ${STATE_TYPE}"
    echo "  Region:         ${STATE_REGION}"
    echo "  Created:        ${STATE_CREATED}"
    echo "  Rate:           \$${price_dollars}/hr"
    echo ""
    echo "  SSH Command:    ssh ${SSH_HOST_ALIAS}"
    echo "  Alt SSH:        ssh ${SSH_USER}@${STATE_IP}"
    echo "  VS Code:        code --remote ssh-remote+${SSH_HOST_ALIAS} /home/${SSH_USER}/workspace/ingero"
    echo ""
    echo "  State File:     ${STATE_FILE}"
    echo ""

    if [[ "$live_status" == "active" ]]; then
        print_warn "Billing is active at \$${price_dollars}/hr."
        print_info "Destroy when done: $0 destroy"
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    local command="${1:-help}"
    shift || true

    # Parse --force / -y flag
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
        destroy)  cmd_destroy "$@" ;;
        status)   cmd_status "$@" ;;
        ssh)      cmd_ssh "$@" ;;
        info)     cmd_info "$@" ;;
        help|-h|--help)
            echo "Usage: $0 <command> [--force|-y]"
            echo ""
            echo "Commands:"
            echo "  deploy    Launch a Lambda Labs GPU instance"
            echo "  destroy   Terminate the instance (stops billing)"
            echo "  status    Show current instance status"
            echo "  ssh       Open SSH session to the instance"
            echo "  info      Show connection details and cost info"
            echo ""
            echo "Flags:"
            echo "  --force, -y  Skip confirmation prompts"
            echo ""
            echo "Note: Lambda Labs does NOT support stop/resume."
            echo "      Billing runs continuously from deploy to destroy."
            echo ""
            echo "Environment:"
            echo "  LAMBDALABS_API_TOKEN  API key (or set in .env file)"
            echo "  Instance types: ${INSTANCE_TYPES[*]}"
            echo ""
            echo "Workflow: deploy -> use -> destroy (no pause)"
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Run '$0 help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
