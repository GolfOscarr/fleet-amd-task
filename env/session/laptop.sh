#!/usr/bin/env bash
# The laptop side of a session (docs/round-2/01-preparation.md, P4; docs/gpu-bringup/06-agent-guide.md).
#
#   bash env/session/laptop.sh balance              # team page: balance, rate, runout, VMs
#   bash env/session/laptop.sh provision            # one VM from the TUI (refuses if one exists); then ip
#   bash env/session/laptop.sh ip                   # the VM's address, saved to env/session/vm.ip
#   bash env/session/laptop.sh push                 # rsync the tree to the VM (absolute destination)
#   bash env/session/laptop.sh login                # docker login ghcr.io on the VM with the laptop's gh token
#   bash env/session/laptop.sh start <stage> [args] # vm.sh start <stage> on the VM
#   bash env/session/laptop.sh status               # vm.sh status on the VM
#   bash env/session/laptop.sh ssh <command...>     # a command on the VM, stdin closed
#   bash env/session/laptop.sh pull                 # logs and record back (absolute paths), then a commit
#   bash env/session/laptop.sh delete --yes         # delete the VM from the TUI and verify the rate is $0.00
#
# DRY=1 prints the commands. Never run two of these at once against the TUI.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TUI="${TUI:-python3 $ROOT/env/hotaisle/tui.py}"
VM_IP_FILE="${VM_IP_FILE:-$ROOT/env/session/vm.ip}"
REMOTE_USER="${REMOTE_USER:-hotaisle}"
REMOTE_DIR="${REMOTE_DIR:-/home/hotaisle/metalOps}"
DRY="${DRY:-0}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
EXCLUDES=(--exclude .venv --exclude .venv-fleet --exclude env/hw/build --exclude env/hw/probes/work
          --exclude env/offline_gfx942/work --exclude docs/report --exclude .omc --exclude harness/fleet_out
          --exclude env/logs --exclude '__pycache__' --exclude env/session/vm.ip --exclude '*.safetensors'
          --exclude repos/fleet-chiplet-megakernel/build --exclude repos/fleet-chiplet-megakernel/permanent_output_dir)

run() { if [ "$DRY" = "1" ]; then echo "+ $*"; return 0; fi; "$@"; }
ip() { [ -f "$VM_IP_FILE" ] && cat "$VM_IP_FILE"; }
need_ip() { local i; i="$(ip)"; [ -n "$i" ] || { echo "no VM address: run laptop.sh ip"; exit 1; }; echo "$i"; }
vm() { local i; i="$(need_ip)"; run ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$i" "$@" </dev/null; }

cmd_balance() {
  # shellcheck disable=SC2086
  run $TUI 12 | grep -E "Available Balance|Hourly Rate|Estimated Runout|No virtual machines|gpuvm|MI300X" || true
}

cmd_provision() {
  local page
  # shellcheck disable=SC2086
  page="$(run $TUI 12)"
  if [ "$DRY" != "1" ] && ! echo "$page" | grep -q "No virtual machines"; then
    echo "a VM seems to exist already (team page below); not provisioning"; echo "$page"; return 1
  fi
  # shellcheck disable=SC2086
  run $TUI 8 n WAIT5 ENTER WAIT8 y WAIT25 | tail -20
  sleep 15
  cmd_ip
}

cmd_ip() {
  local out addr
  # shellcheck disable=SC2086
  out="$(run $TUI 10 ENTER WAIT8)"
  addr="$(echo "$out" | grep -oE 'ssh hotaisle@[0-9.]+' | head -1 | sed 's/ssh hotaisle@//')"
  [ "$DRY" = "1" ] && addr="${addr:-0.0.0.0}"
  [ -n "$addr" ] || { echo "no 'ssh hotaisle@<ip>' on the VM page:"; echo "$out" | tail -20; return 1; }
  echo "$addr" > "$VM_IP_FILE"; echo "$addr"
}

cmd_push() {
  local i; i="$(need_ip)"
  vm "mkdir -p $REMOTE_DIR"
  run rsync -az "${EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" "$ROOT/" "$REMOTE_USER@$i:$REMOTE_DIR/"
}

cmd_login() {
  local i user; i="$(need_ip)"
  user="${GHCR_USER:-$(gh api user --jq .login 2>/dev/null)}"
  [ -n "$user" ] || { echo "no GitHub user (gh auth status)"; return 1; }
  if [ "$DRY" = "1" ]; then echo "+ gh auth token | ssh $REMOTE_USER@$i docker login ghcr.io -u $user --password-stdin"; return 0; fi
  gh auth token | ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$i" "docker login ghcr.io -u $user --password-stdin"
}

cmd_start() { vm "cd $REMOTE_DIR && bash env/session/vm.sh start $*"; }
cmd_status() { vm "cd $REMOTE_DIR && bash env/session/vm.sh status"; }
cmd_ssh() { vm "$@"; }

cmd_pull() {
  local i; i="$(need_ip)"
  local day; day="$(date -u +%Y%m%d)"
  vm "cd $REMOTE_DIR && bash env/session/vm.sh snapshot-logs" || true
  mkdir -p "$ROOT/env/logs/vm"
  run rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$i:$REMOTE_DIR/env/logs/" "$ROOT/env/logs/vm/"
  run rsync -az --exclude build --exclude probes/work --exclude '*.safetensors' --max-size=400k \
    -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$i:$REMOTE_DIR/env/hw/" "$ROOT/env/hw/"
  run rsync -az -e "ssh ${SSH_OPTS[*]}" --include '*.json' --include '*.log' --exclude '*' \
    "$REMOTE_USER@$i:$REMOTE_DIR/harness/ref/" "$ROOT/harness/ref/"
  run rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$i:$REMOTE_DIR/env/check_day1.log" "$ROOT/env/check_day1.log" 2>/dev/null || true
  run rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$i:$REMOTE_DIR/fleet/tasks/results/" "$ROOT/fleet/tasks/results/" 2>/dev/null || true
  [ "$DRY" = "1" ] && return 0
  # commit when the pull brings more than logs (a run, the record, reference files); a logs-only
  # pull stays staged for the next commit (the user's rule, docs/round-2/02-session-plan.md)
  (cd "$ROOT" && git add env/hw harness/ref env/check_day1.log fleet/tasks/results 2>/dev/null
   if git diff --cached --name-only | grep -qv "/logs/"; then
     git commit -q -m "record: pull $(date -u +%Y-%m-%dT%H:%MZ) from the VM (env/hw/$day)" && git log --oneline -1
   elif ! git diff --cached --quiet; then echo "logs only: staged, not committed"
   else echo "nothing new to commit"; fi)
}

cmd_delete() {
  [ "${1:-}" = "--yes" ] || { echo "delete needs --yes (the user decides deletions)"; return 2; }
  # shellcheck disable=SC2086
  run $TUI 10 ENTER WAIT6 DOWN DOWN DOWN DOWN DOWN DOWN WAIT2 ENTER WAIT4 y WAIT20 | tail -5
  sleep 5
  local page
  # shellcheck disable=SC2086
  page="$(run $TUI 12)"
  echo "$page" | grep -E "No virtual machines|Hourly Rate" || true
  if [ "$DRY" != "1" ] && ! echo "$page" | grep -q 'Hourly Rate: \$0.00'; then echo "rate is not \$0.00: check the TUI by hand"; return 1; fi
  rm -f "$VM_IP_FILE"
}

case "${1:-}" in
  balance) cmd_balance;;
  provision) cmd_provision;;
  ip) cmd_ip;;
  push) cmd_push;;
  login) cmd_login;;
  start) shift; cmd_start "$@";;
  status) cmd_status;;
  ssh) shift; cmd_ssh "$@";;
  pull) cmd_pull;;
  delete) shift; cmd_delete "$@";;
  *) sed -n 2,16p "$0"; exit 2;;
esac
