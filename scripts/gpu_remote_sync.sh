#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/gpu_remote_sync.sh <action> [user@host] [options]

Actions:
  check           Test SSH connectivity and print the remote working directory.
  push            Sync local worktree to the remote GPU machine.
  pull            Sync remote lab profile results back to this worktree.
  pull-profiles   Alias for pull.
  exec            Run a command on the remote machine from the project root.

Options:
  --host <user@host>       SSH target. Can also be positional or GPU_HOST.
  --port <port>            SSH port. Default: 22 or GPU_PORT.
  --remote-dir <path>      Remote project directory. Default: ai-infra-labs
                           Use a relative path or absolute path; avoid "~".
  --lab <lab_id_or_path>   Lab for pull. Use "all" for every labs/*/profiles.
                           Default: all or GPU_LAB.
  --raw                    Include raw profiler files such as .nsys-rep.
  --ask-pass              Prompt once for SSH password and use sshpass.
  --password <password>   Use password directly. Prefer --ask-pass or SSH_PASS.
  --no-delete             For push, do not delete remote files missing locally.
  -h, --help              Show this help.

Password handling:
  Preferred:
    SSH_PASS='...' bash scripts/gpu_remote_sync.sh push --host root@1.2.3.4 --port 2222

  Safer interactive:
    bash scripts/gpu_remote_sync.sh push --host root@1.2.3.4 --ask-pass

  If no password is provided, ssh/rsync use your normal SSH agent or prompt.
  The wrapper enables SSH connection sharing, so one password prompt is usually
  enough for an action.

Examples:
  bash scripts/gpu_remote_sync.sh push --host root@1.2.3.4 --ask-pass
  bash scripts/gpu_remote_sync.sh push root@1.2.3.4 --ask-pass
  bash scripts/gpu_remote_sync.sh push root@1.2.3.4 --port 2222 --ask-pass

  bash scripts/gpu_remote_sync.sh exec --host root@1.2.3.4 -- \
    bash scripts/run_lab.sh --lab 000-cuda-rmsnorm --type rmsnorm -- --mode benchmark --variant all

  bash scripts/gpu_remote_sync.sh pull --host root@1.2.3.4
  bash scripts/gpu_remote_sync.sh pull --host root@1.2.3.4 --lab 000-cuda-rmsnorm
USAGE
}

ACTION="${1:-}"
if [[ -n "$ACTION" ]]; then
  shift
fi

HOST="${GPU_HOST:-}"
PORT="${GPU_PORT:-22}"
REMOTE_DIR="${GPU_REMOTE_DIR:-ai-infra-labs}"
LAB_DIR="${GPU_LAB:-all}"
ASK_PASS=0
PASSWORD="${SSH_PASS:-}"
DELETE=1
INCLUDE_RAW=0
REMOTE_CMD=()
CONTROL_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:?missing value for --remote-dir}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --lab)
      LAB_DIR="${2:?missing value for --lab}"
      shift 2
      ;;
    --ask-pass)
      ASK_PASS=1
      shift
      ;;
    --password)
      PASSWORD="${2:?missing value for --password}"
      shift 2
      ;;
    --no-delete)
      DELETE=0
      shift
      ;;
    --raw)
      INCLUDE_RAW=1
      shift
      ;;
    --)
      shift
      REMOTE_CMD=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" != -* ]]; then
        HOST="$1"
        shift
      else
        echo "unknown argument: $1" >&2
        echo >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$ACTION" || "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$HOST" ]]; then
  echo "missing SSH target. Pass user@host, --host user@host, or set GPU_HOST." >&2
  exit 1
fi

if [[ "$ASK_PASS" -eq 1 && -z "$PASSWORD" ]]; then
  read -r -s -p "SSH password for $HOST: " PASSWORD
  echo
fi

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [[ "$LAB_DIR" != "all" && "$LAB_DIR" != labs/* ]]; then
  LAB_DIR="labs/$LAB_DIR"
fi

require_sshpass_if_needed() {
  if [[ -n "$PASSWORD" ]] && ! command -v sshpass >/dev/null 2>&1; then
    cat >&2 <<'MSG'
sshpass is required when using --ask-pass, --password, or SSH_PASS.

Install it on the local machine, or omit the password option and use SSH key auth
or your normal interactive SSH password prompt.
MSG
    exit 1
  fi
}

cleanup() {
  if [[ -n "$CONTROL_DIR" && -d "$CONTROL_DIR" ]]; then
    rm -rf "$CONTROL_DIR"
  fi
}

setup_control_dir() {
  CONTROL_DIR="$(mktemp -d "/tmp/gpu-sync.XXXXXX")"
  trap cleanup EXIT
}

ssh_base_args() {
  printf '%s\n' \
    -p "$PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o ControlMaster=auto \
    -o ControlPersist=10m \
    -o "ControlPath=$CONTROL_DIR/%C"
}

rsync_ssh_command() {
  printf 'ssh -p %q -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=%q/%%C' \
    "$PORT" \
    "$CONTROL_DIR"
}

remote_quote() {
  printf "%q" "$1"
}

run_ssh() {
  local remote_script="$1"
  if [[ -n "$PASSWORD" ]]; then
    SSHPASS="$PASSWORD" sshpass -e ssh $(ssh_base_args) "$HOST" "$remote_script"
  else
    ssh $(ssh_base_args) "$HOST" "$remote_script"
  fi
}

run_rsync() {
  if [[ -n "$PASSWORD" ]]; then
    SSHPASS="$PASSWORD" sshpass -e rsync "$@"
  else
    rsync "$@"
  fi
}

push_worktree() {
  local delete_arg=()
  if [[ "$DELETE" -eq 1 ]]; then
    delete_arg=(--delete)
  fi

  run_ssh "mkdir -p $(remote_quote "$REMOTE_DIR")"

  run_rsync \
    -az \
    "${delete_arg[@]}" \
    -e "$(rsync_ssh_command)" \
    --exclude ".git/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".venv/" \
    --exclude "venv/" \
    --exclude "build/" \
    --exclude "dist/" \
    --exclude "*.egg-info/" \
    --exclude "*.so" \
    --exclude "*.o" \
    --exclude "*.out" \
    --exclude "*.qdrep" \
    --exclude "*.nsys-rep" \
    --exclude "*.ncu-rep" \
    --exclude "*.sqlite" \
    --exclude "labs/*/profiles/" \
    "$ROOT/" \
    "$HOST:$REMOTE_DIR/"
}

pull_profiles() {
  local raw_excludes=()
  if [[ "$INCLUDE_RAW" -eq 0 ]]; then
    raw_excludes=(
      --exclude "*.nsys-rep"
      --exclude "*.qdstrm"
      --exclude "*.sqlite"
      --exclude "*.ncu-rep"
      --exclude ".*.nsys-rep.*"
      --exclude ".*.qdstrm.*"
      --exclude ".*.sqlite.*"
      --exclude ".*.ncu-rep.*"
    )
  fi

  if [[ "$LAB_DIR" == "all" ]]; then
    local local_labs="$ROOT/labs"
    local remote_labs="$REMOTE_DIR/labs"
    local remote_profiles

    mkdir -p "$local_labs"
    remote_profiles="$(
      run_ssh "if [ -d $(remote_quote "$remote_labs") ]; then find $(remote_quote "$remote_labs") -mindepth 2 -maxdepth 2 -type d -name profiles -print; fi"
    )"

    if [[ -z "$remote_profiles" ]]; then
      echo "no remote profiles found under $remote_labs"
      return
    fi

    while IFS= read -r remote_profile; do
      [[ -n "$remote_profile" ]] || continue

      local rel="${remote_profile#"$remote_labs"/}"
      local lab_name="${rel%/profiles}"
      local local_profiles="$local_labs/$lab_name/profiles"

      mkdir -p "$local_profiles"
      echo "pulling $remote_profile -> $local_profiles"

      run_rsync \
        -az \
        "${raw_excludes[@]}" \
        -e "$(rsync_ssh_command)" \
        "$HOST:$remote_profile/" \
        "$local_profiles/"
    done <<< "$remote_profiles"
    return
  fi

  local local_profiles="$ROOT/$LAB_DIR/profiles"
  local remote_profiles="$REMOTE_DIR/$LAB_DIR/profiles"

  mkdir -p "$local_profiles"
  run_ssh "mkdir -p $(remote_quote "$remote_profiles")"

  run_rsync \
    -az \
    "${raw_excludes[@]}" \
    -e "$(rsync_ssh_command)" \
    "$HOST:$remote_profiles/" \
    "$local_profiles/"
}

exec_remote() {
  if [[ "${#REMOTE_CMD[@]}" -eq 0 ]]; then
    echo "exec action requires a command after --" >&2
    exit 1
  fi

  local quoted_dir
  quoted_dir="$(remote_quote "$REMOTE_DIR")"

  local quoted_cmd=()
  local arg
  for arg in "${REMOTE_CMD[@]}"; do
    quoted_cmd+=("$(remote_quote "$arg")")
  done

  local command_text
  command_text="${quoted_cmd[*]}"

  run_ssh "cd $quoted_dir && bash -lc $(remote_quote "$command_text")"
}

check_remote() {
  run_ssh "pwd && whoami && hostname"
}

require_sshpass_if_needed
setup_control_dir

case "$ACTION" in
  check)
    check_remote
    ;;
  push)
    push_worktree
    ;;
  pull|pull-remote|pull-profiles|pull-results)
    pull_profiles
    ;;
  exec)
    exec_remote
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
