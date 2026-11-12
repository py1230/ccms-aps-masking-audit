#!/usr/bin/env bash
# 一键核验入口（便携）：重建评价侧材料 -> 加载 30 个模型核验 -> 生成报告 -> 写执行回执。
#
# 用法：
#   bash run_verification.sh                    # 用包内默认路径
#   bash run_verification.sh --data /path/aps_failure_training_set.csv
#
# 依赖：Python 3.10+、numpy、pandas、scikit-learn、joblib（版本见 00_总说明.md）
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"          # 03_本次事后重建与核验
ROOT="$(cd "$PKG/.." && pwd)"          # 交付包根

DATA="$PKG/data/aps_failure_training_set.csv"
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    *) echo "未知参数：$1" >&2; exit 64 ;;
  esac
done

mkdir -p "$PKG/logs" "$PKG/verify_out"
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EP="$(date +%s)"

python3 "$HERE/rebuild_a3_supplement.py" \
  --project-root "$ROOT" --data "$DATA" --out "$PKG/rebuilt" --code "$PKG/code" > "$PKG/logs/rebuild.log" 2>&1
EC1=$?

if [ "$EC1" -ne 0 ]; then
  echo "rebuild 失败，见 $PKG/logs/rebuild.log" >&2
  FINAL=$EC1
else
  python3 "$HERE/verify_a3_models.py" \
    --data "$DATA" \
    --repro "$ROOT/02_11-12复现产物/A3_repro_2026-11-12" \
    --rebuilt "$PKG/rebuilt" --out "$PKG/verify_out" --code "$PKG/code" > "$PKG/logs/verify.log" 2>&1
  EC2=$?
  if [ "$EC2" -ne 0 ]; then
    echo "verify 失败，见 $PKG/logs/verify.log" >&2
    FINAL=$EC2
  else
    python3 "$HERE/make_report.py" \
      --verify "$PKG/verify_out/verification.json" \
      --rebuild "$PKG/rebuilt/rebuild_summary.json" \
      --out "$PKG/verify_out/核验报告.md" >> "$PKG/logs/verify.log" 2>&1
    FINAL=$?
  fi
fi

END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EP="$(date +%s)"
FINAL=${FINAL:-1}

python3 - "$FINAL" "$START" "$END" "$START_EP" "$END_EP" "$PKG" <<'PY' > "$PKG/logs/execution_receipt.json.tmp"
import hashlib, json, os, platform, socket, sys, getpass

ec, start, end, ep0, ep1, pkg = sys.argv[1:7]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


logs = {}
for n in ("rebuild.log", "verify.log"):
    p = os.path.join(pkg, "logs", n)
    logs[n] = {"sha256": sha(p), "bytes": os.path.getsize(p)} if os.path.exists(p) else None

rec = {
    "identity": "本次事后重建与核验（2026-11-12）",
    "entry": "bash 03_本次事后重建与核验/scripts/run_verification.sh",
    "steps": [
        "rebuild_a3_supplement.py -> rebuilt/",
        "verify_a3_models.py -> verify_out/verification.json",
        "make_report.py -> verify_out/核验报告.md",
    ],
    "exit_code": int(ec),
    "exit_code_source": "outer shell captured $? of each python3 step",
    "cwd": pkg,
    "hostname": socket.gethostname(),
    "user": getpass.getuser(),
    "started_at_utc": start,
    "ended_at_utc": end,
    "wall_seconds_os": int(ep1) - int(ep0),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "logs": logs,
    "note": "回执由外层包装器在全部步骤结束后写；不代表科学结论成立。",
}
print(json.dumps(rec, ensure_ascii=False, indent=1))
PY

mv "$PKG/logs/execution_receipt.json.tmp" "$PKG/logs/execution_receipt.json"
echo "EXIT:$FINAL"
echo "report: $PKG/verify_out/核验报告.md"
exit "$FINAL"
