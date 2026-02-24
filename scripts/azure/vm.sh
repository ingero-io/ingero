#!/bin/bash
################################################################################
# Ingero Azure GPU VM Lifecycle Manager
#
# Automates Azure GPU VM creation, start/stop (deallocate), and teardown for
# eBPF development sessions. Azure supports deallocate (stops compute billing,
# keeps disk) — unlike Lambda Labs which only has deploy/destroy.
#
# Prerequisites:
#   - Azure CLI (az) installed and logged in (az login)
#   - AZURE_SUBSCRIPTION_ID set (env var or .env file)
#   - SSH key at ~/.ssh/id_ed25519.pub
#
# Usage:
#   ./scripts/azure/vm.sh deploy    # Create and provision a new GPU VM
#   ./scripts/azure/vm.sh start     # Start a deallocated VM (resumes billing)
#   ./scripts/azure/vm.sh stop      # Deallocate VM (stops compute billing, keeps disk)
#   ./scripts/azure/vm.sh destroy   # Delete resource group permanently (fully stops billing)
#   ./scripts/azure/vm.sh status    # Show VM status
#   ./scripts/azure/vm.sh ssh       # SSH into the VM
#   ./scripts/azure/vm.sh info      # Show connection details and cost info
#
# Flags:
#   --force, -y    Skip confirmation prompts
#
# Workflow:
#   deploy -> use -> destroy when done (VMs are ephemeral)
#   Or: deploy -> use -> stop (pause billing) -> start (resume) -> destroy
#
# State: VM details stored in .azure-vm.json (gitignored)
#
# Auth: Azure CLI cached login (az login). No client_id/secret needed.
# Quota: GPU VM families default to 0 vCPUs — request increase via Azure Portal.
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

# Azure defaults
AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-ingero-gpu-rg}"
AZURE_REGION="${AZURE_REGION:-eastus}"
AZURE_VM_NAME="${AZURE_VM_NAME:-ingero-gpu-dev}"
AZURE_SSH_USER="${AZURE_SSH_USER:-azureuser}"
AZURE_IMAGE="${AZURE_IMAGE:-Canonical:ubuntu-24_04-lts:server:latest}"

# VM size preference order (override with AZURE_VM_SIZE env var for a single size)
# Cheapest viable GPU first: T4 ($0.53/hr) > V100 ($3.06/hr) > A100 ($3.67/hr)
if [[ -n "${AZURE_VM_SIZE:-}" ]]; then
    VM_SIZES=("$AZURE_VM_SIZE")
else
    VM_SIZES=(
        "Standard_NC4as_T4_v3"           # 1x T4 16GB, 4 vCPU, ~$0.53/hr
        "Standard_NC6s_v3"               # 1x V100 16GB, 6 vCPU, ~$3.06/hr
        "Standard_NC24ads_A100_v4"       # 1x A100 80GB, 24 vCPU, ~$3.67/hr
    )
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to mono-repo root (3 levels up from agent/scripts/azure/).
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STATE_FILE="$PROJECT_DIR/.azure-vm.json"
ENV_FILE="$PROJECT_DIR/.env"

# SSH
SSH_KEY_PATH="$HOME/.ssh/id_ed25519"
SSH_HOST_ALIAS="ingero-azure"

# Timeouts
SSH_WAIT_TIMEOUT=600     # 10 minutes — Azure GPU VMs can take 5-8 min to provision
SSH_WAIT_INTERVAL=15     # Poll every 15 seconds
CLOUD_INIT_TIMEOUT=900   # 15 minutes for cloud-init to finish

# ============================================================================
# Utility Functions
# ============================================================================

load_subscription() {
    if [[ -n "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
        return 0
    fi

    if [[ -f "$ENV_FILE" ]]; then
        AZURE_SUBSCRIPTION_ID=$(grep -E '^AZURE_SUBSCRIPTION_ID=' "$ENV_FILE" | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'")
        export AZURE_SUBSCRIPTION_ID
    fi

    if [[ -z "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
        # Try to get from az CLI directly
        AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv 2>/dev/null || true)
        export AZURE_SUBSCRIPTION_ID
    fi

    if [[ -z "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
        print_error "AZURE_SUBSCRIPTION_ID not set and az CLI not logged in."
        print_info "Option 1: az login (then subscription is auto-detected)"
        print_info "Option 2: export AZURE_SUBSCRIPTION_ID=your-sub-id"
        print_info "Option 3: Add to .env: AZURE_SUBSCRIPTION_ID=your-sub-id"
        exit 1
    fi
}

check_prerequisites() {
    local missing=()

    if ! command -v az &>/dev/null; then
        print_warn "Azure CLI (az) not installed. Installing..."
        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
        if ! command -v az &>/dev/null; then
            print_error "Azure CLI installation failed."
            print_info "Manual install: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
            exit 1
        fi
        print_success "Azure CLI installed: $(az version --query '\"azure-cli\"' -o tsv)"
    fi

    if ! command -v jq &>/dev/null; then missing+=("jq"); fi
    if ! command -v ssh &>/dev/null; then missing+=("ssh"); fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing[*]}"
        print_info "Install via: sudo apt install ${missing[*]}"
        exit 1
    fi

    # Verify az is logged in
    if ! az account show &>/dev/null 2>&1; then
        print_error "Azure CLI not logged in."
        print_info "Run: az login"
        exit 1
    fi

    load_subscription

    local ssh_pub_key_path="${SSH_KEY_PATH}.pub"
    if [[ ! -f "$ssh_pub_key_path" ]]; then
        print_error "SSH public key not found: $ssh_pub_key_path"
        print_info "Generate one: ssh-keygen -t ed25519"
        exit 1
    fi
}

# ============================================================================
# State Management
# ============================================================================

save_state() {
    local resource_group="$1"
    local vm_name="$2"
    local ip="$3"
    local vm_size="$4"
    local region="$5"
    local status="$6"

    local created
    if [[ -f "$STATE_FILE" ]]; then
        created=$(jq -r '.created // empty' "$STATE_FILE" 2>/dev/null)
    fi
    created="${created:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

    jq -n \
        --arg rg "$resource_group" \
        --arg name "$vm_name" \
        --arg ip "$ip" \
        --arg size "$vm_size" \
        --arg region "$region" \
        --arg status "$status" \
        --arg created "$created" \
        '{resource_group: $rg, vm_name: $name, ip: $ip, vm_size: $size, region: $region, status: $status, created: $created}' \
        > "$STATE_FILE"
}

load_state() {
    if [[ ! -f "$STATE_FILE" ]]; then
        return 1
    fi

    STATE_RG=$(jq -r '.resource_group' "$STATE_FILE")
    STATE_VM_NAME=$(jq -r '.vm_name' "$STATE_FILE")
    STATE_IP=$(jq -r '.ip' "$STATE_FILE")
    STATE_VM_SIZE=$(jq -r '.vm_size' "$STATE_FILE")
    STATE_REGION=$(jq -r '.region' "$STATE_FILE")
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
    local ssh_config="$HOME/.ssh/config"

    mkdir -p "$HOME/.ssh"
    touch "$ssh_config"

    # Remove existing block if present
    if grep -q "# BEGIN ingero-azure" "$ssh_config" 2>/dev/null; then
        local temp_file="${ssh_config}.tmp"
        sed '/# BEGIN ingero-azure/,/# END ingero-azure/d' "$ssh_config" > "$temp_file"
        mv "$temp_file" "$ssh_config"
    fi

    # Append managed block — Azure uses port 22, user 'azureuser'
    cat >> "$ssh_config" << EOF

# BEGIN ingero-azure (managed by azure/vm.sh - do not edit manually)
Host ${SSH_HOST_ALIAS}
    HostName ${ip}
    Port 22
    User ${AZURE_SSH_USER}
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
# END ingero-azure
EOF

    print_success "SSH config updated: ssh ${SSH_HOST_ALIAS}"
}

remove_ssh_config() {
    local ssh_config="$HOME/.ssh/config"

    if [[ -f "$ssh_config" ]] && grep -q "# BEGIN ingero-azure" "$ssh_config" 2>/dev/null; then
        local temp_file="${ssh_config}.tmp"
        sed '/# BEGIN ingero-azure/,/# END ingero-azure/d' "$ssh_config" > "$temp_file"
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

    print_info "Waiting for VM to become SSH-ready (timeout: ${SSH_WAIT_TIMEOUT}s)..."

    while [[ $elapsed -lt $SSH_WAIT_TIMEOUT ]]; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -i "$SSH_KEY_PATH" "${AZURE_SSH_USER}@${ip}" "echo ready" 2>/dev/null; then
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
# Cloud-Init
# ============================================================================

generate_cloud_init() {
    # Returns cloud-init YAML that installs all Ingero dependencies.
    # Same package set as TensorDock/Lambda cloud-init scripts.
    cat << 'CLOUDINIT'
#cloud-config
package_update: true
packages:
  - build-essential
  - clang-14
  - llvm-14
  - libbpf-dev
  - libelf-dev
  - zlib1g-dev
  - linux-tools-common
  - linux-tools-generic
  - make
  - pkg-config
  - git
  - wget
  - curl
  - vim
  - jq
  - sqlite3
  - python3-pip
  - stress-ng

runcmd:
  - apt-get install -y -qq linux-tools-$(uname -r) 2>/dev/null || true
  - apt-get install -y -qq bpftrace trace-cmd 2>/dev/null || true
  # Go 1.26.0
  - cd /tmp && wget -q https://go.dev/dl/go1.26.0.linux-amd64.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.26.0.linux-amd64.tar.gz && rm go1.26.0.linux-amd64.tar.gz
  - grep -q /usr/local/go/bin /home/azureuser/.bashrc || echo 'export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/go/bin:$HOME/.local/bin:$PATH' >> /home/azureuser/.bashrc
  # PyTorch + ML deps
  - sudo -u azureuser pip3 install --quiet torch torchvision numpy --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3 || true
  - sudo -u azureuser pip3 install --quiet transformers datasets diffusers accelerate 2>&1 | tail -3 || true
  # Workspace
  - mkdir -p /home/azureuser/workspace
  - chown -R azureuser:azureuser /home/azureuser/workspace
  # Signal completion
  - echo "Cloud-init complete: $(date)" > /home/azureuser/workspace/cloud-init-done.txt
  - chown azureuser:azureuser /home/azureuser/workspace/cloud-init-done.txt
CLOUDINIT
}

# ============================================================================
# Commands
# ============================================================================

cmd_deploy() {
    print_header "Deploying Ingero GPU VM (Azure)"

    # Idempotency guard
    if load_state; then
        print_error "VM already exists: ${STATE_VM_NAME} in ${STATE_RG}"
        print_info "Status: ${STATE_STATUS} | IP: ${STATE_IP}"
        print_info "Use '$0 destroy' first, or '$0 start' to resume."
        exit 1
    fi

    check_prerequisites

    # Confirm
    print_warn "Cost Warning:"
    print_info "  VM sizes (preference order): ${VM_SIZES[*]}"
    print_info "  Region: ${AZURE_REGION}"
    print_info "  Image: ${AZURE_IMAGE}"
    print_info "  Resource group: ${AZURE_RESOURCE_GROUP}"
    print_info "  Billing starts immediately upon deployment."
    print_info "  Deallocate (stop billing): $0 stop"
    print_info "  Fully destroy: $0 destroy"
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

    # Set subscription
    print_info "Setting subscription: ${AZURE_SUBSCRIPTION_ID}"
    az account set --subscription "$AZURE_SUBSCRIPTION_ID"

    # Create resource group
    print_info "Creating resource group: ${AZURE_RESOURCE_GROUP} in ${AZURE_REGION}..."
    az group create --name "$AZURE_RESOURCE_GROUP" --location "$AZURE_REGION" --output none

    # Write cloud-init to temp file
    local cloud_init_file
    cloud_init_file=$(mktemp /tmp/ingero-cloud-init-XXXXXX.yaml)
    generate_cloud_init > "$cloud_init_file"

    # Try VM sizes in preference order
    local deployed_size=""
    local create_output=""

    for vm_size in "${VM_SIZES[@]}"; do
        print_info "Trying VM size: ${vm_size}..."

        create_output=$(az vm create \
            --resource-group "$AZURE_RESOURCE_GROUP" \
            --name "$AZURE_VM_NAME" \
            --image "$AZURE_IMAGE" \
            --size "$vm_size" \
            --admin-username "$AZURE_SSH_USER" \
            --ssh-key-values "${SSH_KEY_PATH}.pub" \
            --custom-data "$cloud_init_file" \
            --public-ip-sku Standard \
            --security-type Standard \
            --output json 2>&1) && {
            deployed_size="$vm_size"
            break
        }

        # Check if quota error
        if echo "$create_output" | grep -qi "quota\|QuotaExceeded\|OperationNotAllowed\|capacity"; then
            print_warn "  ${vm_size}: quota/capacity unavailable in ${AZURE_REGION}"
            # Clean up failed VM if partially created
            az vm delete --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_VM_NAME" --yes --no-wait 2>/dev/null || true
            continue
        fi

        print_warn "  ${vm_size}: failed — $(echo "$create_output" | head -3)"
        az vm delete --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_VM_NAME" --yes --no-wait 2>/dev/null || true
    done

    # Clean up temp file
    rm -f "$cloud_init_file"

    if [[ -z "$deployed_size" ]]; then
        print_error "Failed to create VM with any available size."
        print_info "Sizes tried: ${VM_SIZES[*]}"
        print_info "Region: ${AZURE_REGION}"
        print_info ""
        print_info "Common fix: Request GPU quota increase in Azure Portal:"
        print_info "  Portal -> Subscriptions -> Usage + quotas"
        print_info "  Search for NC/ND families, request vCPU increase"
        print_info ""
        print_info "Or try a different region: AZURE_REGION=westus2 $0 deploy"
        # Clean up empty resource group
        az group delete --name "$AZURE_RESOURCE_GROUP" --yes --no-wait 2>/dev/null || true
        exit 1
    fi

    print_success "VM created: ${AZURE_VM_NAME} (${deployed_size})"

    # Extract public IP
    local ip
    ip=$(echo "$create_output" | jq -r '.publicIpAddress // empty' 2>/dev/null)

    if [[ -z "$ip" || "$ip" == "null" ]]; then
        # Fallback: query IP directly
        print_info "Querying public IP..."
        ip=$(az vm show -d --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_VM_NAME" --query publicIps -o tsv 2>/dev/null)
    fi

    if [[ -z "$ip" || "$ip" == "null" ]]; then
        print_error "Could not determine VM public IP."
        save_state "$AZURE_RESOURCE_GROUP" "$AZURE_VM_NAME" "" "$deployed_size" "$AZURE_REGION" "unknown"
        print_info "State saved. Check: $0 status"
        exit 1
    fi

    # Open port 22 (NSG rule) — az vm create usually does this, but ensure it
    print_info "Ensuring SSH port is open..."
    az vm open-port --port 22 --resource-group "$AZURE_RESOURCE_GROUP" --name "$AZURE_VM_NAME" --output none 2>/dev/null || true

    # Save state
    save_state "$AZURE_RESOURCE_GROUP" "$AZURE_VM_NAME" "$ip" "$deployed_size" "$AZURE_REGION" "running"
    print_success "State saved to ${STATE_FILE}"

    # Update SSH config
    update_ssh_config "$ip"

    # Wait for SSH
    wait_for_ssh "$ip" || true

    # Pricing info
    local price_info=""
    case "$deployed_size" in
        *T4*)   price_info="\$0.53/hr (T4 16GB)" ;;
        *NC6s*) price_info="\$3.06/hr (V100 16GB)" ;;
        *A100*) price_info="\$3.67/hr (A100 80GB)" ;;
        *)      price_info="see Azure pricing" ;;
    esac

    # Final output
    print_header "Deployment Complete"
    echo ""
    print_success "VM Details:"
    echo "  VM Name:        ${AZURE_VM_NAME}"
    echo "  Resource Group:  ${AZURE_RESOURCE_GROUP}"
    echo "  IP:             ${ip}"
    echo "  VM Size:        ${deployed_size}"
    echo "  Region:         ${AZURE_REGION}"
    echo "  Rate:           ${price_info}"
    echo ""
    print_success "Connect:"
    echo "  ssh ${SSH_HOST_ALIAS}"
    echo "  # or: ssh ${AZURE_SSH_USER}@${ip}"
    echo ""
    print_warn "Cost Reminder:"
    echo "  Billing is active at ${price_info}."
    echo "  Deallocate (stop compute billing): $0 stop"
    echo "  Destroy (fully stop all billing):  $0 destroy"
    echo ""
    print_info "Cloud-init is installing dependencies (Go, clang, PyTorch, libbpf)."
    print_info "Check if done: ssh ${SSH_HOST_ALIAS} 'cat ~/workspace/cloud-init-done.txt 2>/dev/null || echo still running'"
    print_info "Next steps after cloud-init:"
    print_info "  1. Sync code:  make azure-sync"
    print_info "  2. Validate:   make azure-validate"
    print_info "  3. Or SSH in:  ssh ${SSH_HOST_ALIAS}"
}

cmd_start() {
    print_header "Starting Ingero GPU VM (Azure)"

    if ! load_state; then
        print_error "No VM found. Deploy first: $0 deploy"
        exit 1
    fi

    load_subscription
    az account set --subscription "$AZURE_SUBSCRIPTION_ID"

    print_info "Starting VM: ${STATE_VM_NAME} in ${STATE_RG}..."
    az vm start --resource-group "$STATE_RG" --name "$STATE_VM_NAME"

    # Get the new public IP (may change after deallocate/start)
    print_info "Querying public IP..."
    local ip
    ip=$(az vm show -d --resource-group "$STATE_RG" --name "$STATE_VM_NAME" --query publicIps -o tsv 2>/dev/null)

    if [[ -n "$ip" && "$ip" != "null" ]]; then
        STATE_IP="$ip"
    fi

    # Update state and SSH config
    save_state "$STATE_RG" "$STATE_VM_NAME" "$STATE_IP" "$STATE_VM_SIZE" "$STATE_REGION" "running"
    update_ssh_config "$STATE_IP"

    # Wait for SSH
    wait_for_ssh "$STATE_IP" || true

    print_success "VM started. Connect: ssh ${SSH_HOST_ALIAS}"
    print_warn "Billing is now active."
}

cmd_stop() {
    print_header "Deallocating Ingero GPU VM (Azure)"

    if ! load_state; then
        print_error "No VM found. Nothing to stop."
        exit 1
    fi

    load_subscription
    az account set --subscription "$AZURE_SUBSCRIPTION_ID"

    print_info "Deallocating stops compute billing. Disk storage is still billed."
    print_info "To fully stop all billing, use: $0 destroy"
    if [[ "$FORCE_MODE" != "true" ]]; then
        read -p "Deallocate VM ${STATE_VM_NAME}? [y/N] " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            print_info "Cancelled."
            exit 0
        fi
    else
        print_info "Auto-confirmed (--force)"
    fi

    print_info "Deallocating VM: ${STATE_VM_NAME}..."
    az vm deallocate --resource-group "$STATE_RG" --name "$STATE_VM_NAME"

    save_state "$STATE_RG" "$STATE_VM_NAME" "$STATE_IP" "$STATE_VM_SIZE" "$STATE_REGION" "deallocated"

    print_success "VM deallocated. Compute billing stopped (disk storage still billed)."
    print_info "Resume later: $0 start"
    print_info "Fully stop billing: $0 destroy"
}

cmd_destroy() {
    print_header "Destroying Ingero GPU VM (Azure)"

    if ! load_state; then
        print_error "No VM found. Nothing to destroy."
        exit 1
    fi

    load_subscription
    az account set --subscription "$AZURE_SUBSCRIPTION_ID"

    print_error "WARNING: This will delete the entire resource group '${STATE_RG}'"
    print_error "         including VM, disks, NIC, NSG, and public IP!"
    print_info "VM: ${STATE_VM_NAME}"
    print_info "IP: ${STATE_IP}"
    print_info "Size: ${STATE_VM_SIZE}"
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

    print_info "Deleting resource group: ${STATE_RG} (this cleans up all resources)..."
    az group delete --name "$STATE_RG" --yes --no-wait

    remove_ssh_config
    clear_state

    print_success "Resource group deletion initiated. All billing will stop."
    print_info "Deletion runs in the background (--no-wait). Resources will be removed shortly."
}

cmd_status() {
    if ! load_state; then
        print_info "No VM deployed. Deploy one: $0 deploy"
        exit 0
    fi

    load_subscription
    az account set --subscription "$AZURE_SUBSCRIPTION_ID"

    print_info "Fetching live status for ${STATE_VM_NAME}..."

    local live_status
    live_status=$(az vm get-instance-view \
        --resource-group "$STATE_RG" \
        --name "$STATE_VM_NAME" \
        --query "instanceView.statuses[1].displayStatus" \
        -o tsv 2>/dev/null || echo "unknown")

    # Normalize status
    local normalized_status="$live_status"
    case "$live_status" in
        "VM running")       normalized_status="running" ;;
        "VM deallocated")   normalized_status="deallocated" ;;
        "VM stopped")       normalized_status="stopped" ;;
        "VM starting")      normalized_status="starting" ;;
        "VM deallocating")  normalized_status="deallocating" ;;
    esac

    # Update IP if running
    local live_ip="$STATE_IP"
    if [[ "$normalized_status" == "running" ]]; then
        live_ip=$(az vm show -d --resource-group "$STATE_RG" --name "$STATE_VM_NAME" --query publicIps -o tsv 2>/dev/null || echo "$STATE_IP")
    fi

    # Update local state
    save_state "$STATE_RG" "$STATE_VM_NAME" "$live_ip" "$STATE_VM_SIZE" "$STATE_REGION" "$normalized_status"

    print_header "Ingero GPU VM Status (Azure)"
    echo "  VM Name:         ${STATE_VM_NAME}"
    echo "  Resource Group:  ${STATE_RG}"
    echo "  Status:          ${live_status}"
    echo "  IP:              ${live_ip}"
    echo "  VM Size:         ${STATE_VM_SIZE}"
    echo "  Region:          ${STATE_REGION}"
    echo "  Created:         ${STATE_CREATED}"
    echo ""

    if [[ "$normalized_status" == "running" ]]; then
        print_success "VM is running. Connect: ssh ${SSH_HOST_ALIAS}"
        print_warn "Billing is active. Destroy when done: $0 destroy"
    elif [[ "$normalized_status" == "deallocated" ]]; then
        print_info "VM is deallocated. Compute billing stopped (disk still billed)."
        print_info "Resume: $0 start | Fully stop billing: $0 destroy"
    elif [[ "$normalized_status" == "stopped" ]]; then
        print_warn "VM is stopped (NOT deallocated — compute billing may still be active!)."
        print_info "Deallocate: $0 stop | Fully stop billing: $0 destroy"
    else
        print_warn "VM status: ${live_status}"
    fi
}

cmd_ssh() {
    if ! load_state; then
        print_error "No VM found. Deploy first: $0 deploy"
        exit 1
    fi

    exec ssh -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${AZURE_SSH_USER}@${STATE_IP}" "$@"
}

cmd_info() {
    if ! load_state; then
        print_info "No VM deployed."
        exit 0
    fi

    # Pricing info
    local price_info=""
    case "$STATE_VM_SIZE" in
        *T4*)   price_info="\$0.53/hr (T4 16GB)" ;;
        *NC6s*) price_info="\$3.06/hr (V100 16GB)" ;;
        *A100*) price_info="\$3.67/hr (A100 80GB)" ;;
        *)      price_info="see Azure pricing" ;;
    esac

    print_header "Ingero GPU VM Info (Azure)"
    echo ""
    echo "  VM Name:         ${STATE_VM_NAME}"
    echo "  Resource Group:  ${STATE_RG}"
    echo "  Status:          ${STATE_STATUS}"
    echo "  IP Address:      ${STATE_IP}"
    echo "  VM Size:         ${STATE_VM_SIZE}"
    echo "  Region:          ${STATE_REGION}"
    echo "  Created:         ${STATE_CREATED}"
    echo "  Rate:            ${price_info}"
    echo ""
    echo "  SSH Command:     ssh ${SSH_HOST_ALIAS}"
    echo "  Alt SSH:         ssh ${AZURE_SSH_USER}@${STATE_IP}"
    echo "  VS Code:         code --remote ssh-remote+${SSH_HOST_ALIAS} /home/${AZURE_SSH_USER}/workspace/ingero"
    echo ""
    echo "  State File:      ${STATE_FILE}"
    echo "  Subscription:    ${AZURE_SUBSCRIPTION_ID:-<run az account show>}"
    echo ""

    if [[ "$STATE_STATUS" == "running" ]]; then
        print_warn "Billing is active at ${price_info}."
        print_info "Deallocate (stop compute billing): $0 stop"
        print_info "Destroy (fully stop billing): $0 destroy"
    elif [[ "$STATE_STATUS" == "deallocated" ]]; then
        print_info "VM is deallocated. Disk storage still billed."
        print_info "Resume: $0 start | Destroy: $0 destroy"
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
            echo "  deploy    Create and provision a new Azure GPU VM"
            echo "  start     Start a deallocated VM (resumes compute billing)"
            echo "  stop      Deallocate VM (stops compute billing, disk still billed)"
            echo "  destroy   Delete resource group permanently (fully stops billing)"
            echo "  status    Show current VM status"
            echo "  ssh       Open SSH session to the VM"
            echo "  info      Show connection details and cost info"
            echo ""
            echo "Flags:"
            echo "  --force, -y  Skip confirmation prompts"
            echo ""
            echo "Workflow: deploy -> use -> destroy when done (VMs are ephemeral)"
            echo "  Or:     deploy -> use -> stop (pause billing) -> start -> destroy"
            echo ""
            echo "Environment:"
            echo "  AZURE_SUBSCRIPTION_ID  Subscription ID (or set in .env, or auto-detect via az)"
            echo "  AZURE_REGION           Region (default: eastus)"
            echo "  AZURE_VM_SIZE          Override VM size (default: tries T4, V100, A100)"
            echo "  AZURE_RESOURCE_GROUP   Resource group name (default: ingero-gpu-rg)"
            echo "  AZURE_VM_NAME          VM name (default: ingero-gpu-dev)"
            echo "  AZURE_SSH_USER         SSH username (default: azureuser)"
            echo ""
            echo "GPU Quota:"
            echo "  Azure defaults GPU VM families to 0 vCPUs."
            echo "  Request increase: Portal -> Subscriptions -> Usage + quotas"
            echo "  Search for NC/ND families in your target region."
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Run '$0 help' for usage."
            exit 1
            ;;
    esac
}

main "$@"
