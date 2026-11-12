#!/usr/bin/env bash
# 生产执行包装器：调用主脚本并在结束后写一份执行回执（真实的 argv、cwd、退出码、时间）。
# 回执由外层 shell 记录 $?，不是程序自打印的文本。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 90

LOG="$ROOT/logs/run.log"
RECEIPT="$ROOT/logs/execution_receipt.json"
mkdir -p "$ROOT/logs"

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

python3 "$ROOT/scripts/run_secom_boot20k.py" > "$LOG" 2>&1
EC=$?

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"

python3 - "$EC" "$START_UTC" "$END_UTC" "$START_EPOCH" "$END_EPOCH" "$ROOT" "$LOG" "$RECEIPT" <<'PY' > "$RECEIPT.tmp"
import hashlib, json, os, platform, socket, sys, getpass, subprocess

ec, start_utc, end_utc, start_ep, end_ep, root, log, receipt = sys.argv[1:9]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


script = os.path.join(root, "scripts", "run_secom_boot20k.py")
wrapper = os.path.join(root, "scripts", "run_with_receipt.sh")
rec = {
    "identity": "本次重跑（re-run, 2026-11-12）；不是 2026-11-08 那批 20,000 次的原件",
    "outer_command": "bash scripts/run_with_receipt.sh",
    "inner_command": "python3 scripts/run_secom_boot20k.py",
    "exit_code": int(ec),
    "exit_code_source": "outer shell captured $? of the python3 process",
    "cwd": root,
    "hostname": socket.gethostname(),
    "user": getpass.getuser(),
    "pid_shell": os.getppid(),
    "started_at_utc": start_utc,
    "ended_at_utc": end_utc,
    "wall_seconds_os": int(end_ep) - int(start_ep),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "shell": os.environ.get("SHELL", "bash"),
    "script_sha256": sha(script),
    "wrapper_sha256": sha(wrapper),
    "log_path": log,
    "log_sha256": sha(log),
    "log_lines": sum(1 for _ in open(log, "rb")),
    "last_log_line": open(log, "rb").read().decode("utf-8", "replace").strip().splitlines()[-1],
    "git_commit": None,
    "note": "回执由外层包装器在进程结束后写入；不代表科学结论成立。",
}
print(json.dumps(rec, ensure_ascii=False, indent=1))
PY

mv "$RECEIPT.tmp" "$RECEIPT"
echo "EXIT:$EC"
echo "receipt: $RECEIPT"
exit $EC
