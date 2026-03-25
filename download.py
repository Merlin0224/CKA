import os
from modelscope import snapshot_download

# 目标路径
model_dir = "/root/autodl-tmp/.cache/modelscope/hub/models/qwen/Qwen3-VL-8B-Instruct"

# 开始下载
# 如果目录已存在，snapshot_download 会自动检查完整性，不会重复下载
model_id = "Qwen/Qwen3-VL-8B-Instruct"
snapshot_download(model_id, local_dir=model_dir)

print(f"模型已成功下载至: {model_dir}")