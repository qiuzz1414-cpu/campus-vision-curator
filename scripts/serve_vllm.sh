#!/usr/bin/env bash
# 一键启动本地 vLLM 视觉语言模型服务（OpenAI 兼容端点）
#
# 前置：pip install vllm
# 用法：bash scripts/serve_vllm.sh [模型名]
#
# 默认模型改了之后，记得同步 recipe 里的 label.model

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
# 默认只监听本机：vLLM 的 OpenAI 兼容端点**没有鉴权**，
# 绑 0.0.0.0 等于把裸端点暴露到整个局域网（校园网/宿舍网尤其危险）。
# 确需跨机器访问时显式指定：VLLM_HOST=0.0.0.0 bash scripts/serve_vllm.sh
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"

# 显存不足时按顺序降级：
#   Qwen/Qwen2.5-VL-7B-Instruct      ~16GB
#   Qwen/Qwen2.5-VL-3B-Instruct      ~8GB
#   Qwen/Qwen2-VL-2B-Instruct        ~5GB

echo "启动 vLLM：${MODEL} @ ${HOST}:${PORT}"
echo "服务就绪后，另开终端执行：cvc doctor"
echo

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=4 \
  --served-model-name "$(basename "${MODEL}")"
