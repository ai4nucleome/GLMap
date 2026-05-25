#!/usr/bin/env bash
# 从列表文件逐行读取 Hugging Face 模型 repo_id，调用 hf download 下载。
# 依赖: pip install -U huggingface_hub（提供 hf 命令），需已登录或有 token（见 HF_TOKEN）。
#
# 用法:
#   ./download_models_from_list.sh
#   ./download_models_from_list.sh /path/to/list.txt
#
# 环境变量（可选）:
#   HF_HOME      缓存根目录，默认 ~/.cache/huggingface（模型落在 $HF_HOME/hub）
#
# 若需额外参数（如 --local-dir），可在下面 hf download 一行自行追加，或包一层函数。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST_FILE="${1:-${SCRIPT_DIR}/../../models/download_models_list.txt}"
HF_HOME="${HF_HOME:-/data/yusen/software/.cache/huggingface}"
CACHE_DIR="${HF_HOME}/hub"

if ! command -v hf >/dev/null 2>&1; then
  echo "错误: 未找到命令 'hf'。请先安装: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "错误: 列表文件不存在: $LIST_FILE" >&2
  exit 1
fi

mapfile -t lines < <(grep -vE '^\s*(#|$)' "$LIST_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
if [[ ${#lines[@]} -eq 0 ]]; then
  echo "列表为空或仅含注释/空行: $LIST_FILE" >&2
  exit 1
fi

echo "列表文件: $LIST_FILE"
echo "模型数: ${#lines[@]}"
echo "缓存目录: $CACHE_DIR"
echo ""

failed=()
n=0
for repo_id in "${lines[@]}"; do
  n=$((n + 1))
  echo ">>> [$n/${#lines[@]}] hf download ${repo_id} --exclude *.h5 tf_* *.joblib *.pt *.bin"
  #! 修改这里
  if hf download "$repo_id" --exclude *.h5 tf_* *.joblib *.pt *.bin; then
    echo "    完成: $repo_id"
  else
    echo "    失败: $repo_id" >&2
    failed+=("$repo_id")
  fi
  echo ""
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "以下模型下载失败 (${#failed[@]}):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi

echo "全部下载成功。"
