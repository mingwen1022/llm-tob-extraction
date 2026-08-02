#!/usr/bin/env bash
# 单请求延迟基准：**严格串行**跑，一次只有一个请求在飞。
#
# 为什么要单独跑：主实验的 few-shot 矩阵是 9~16 个进程并发打同一个端点的，
# 测出来的耗时含排队与限流，不能用来做跨模型的延迟比较。这里每次只起一个进程、
# 且模型之间也不重叠，得到的才是干净的单请求延迟。
#
# 样本取每个配置前 N 条（默认 20）——所有模型看同一批文档，长度分布一致，可比。
set -u
GMI="https://api.gmi-serving.com/v1"
N="${1:-20}"
OUT=runs/latency_bench
mkdir -p "$OUT"

run() {
  local model="$1" shots="$2" extra="${3:-}"
  local slug; slug=$(echo "$model" | tr '/.' '__')
  local out="$OUT/${slug}_s${shots}.jsonl"
  [ -s "$out" ] && { echo "[skip] $model s=$shots"; return; }
  echo "[run ] $model s=$shots ..."
  # shellcheck disable=SC2086
  uv run python -u scripts/run_api.py --model "$model" \
      --base-url "$GMI" --api-key-env GMI_API_KEY --full-prompt \
      --shots "$shots" --limit "$N" --max-tokens 4096 --timeout 300 --retries 2 \
      ${extra} --out "$out" > "$OUT/${slug}_s${shots}.log" 2>&1
  echo "[done] $(tail -2 "$OUT/${slug}_s${shots}.log" | head -1)"
}

for shots in 0 16; do
  run "google/gemini-3.5-flash"     "$shots"
  run "MiniMaxAI/MiniMax-M3"        "$shots"
  run "zai-org/GLM-5.2-FP8"         "$shots"
  run "deepseek-ai/DeepSeek-V4-Pro" "$shots"
  run "moonshotai/kimi-k3"          "$shots" "--temperature 1"
  run "Qwen/Qwen3.7-Max"            "$shots"
done
echo "=== 串行延迟基准跑完 ==="
