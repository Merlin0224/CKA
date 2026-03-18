import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from modelscope import snapshot_download
from PIL import Image

# ================= 1. 信息瓶颈层 =================
class BottleneckProjector(nn.Module):
    def __init__(self, in_features, out_features, reduction_ratio=4):
        super().__init__()
        bottleneck_dim = out_features // reduction_ratio

        # 降维 -> 激活 -> 升维 (经典的 Information Bottleneck 结构)
        self.net = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim, bias=True),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_features, bias=True)
        )
    
    def forward(self, x):
        return self.net(x)



# ================= 2. 加载模型 =================
model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained(model_dir)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("正在加载模型...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)


# ================= 3. 动态寻找并替换视觉投影层 =================
target_mlp_name = None
target_mlp_module = None
# 寻找 Qwen2-VL 特有的 'merger.mlp'
for name, module in model.named_modules():
    if 'merger.mlp' in name:
        target_mlp_name = name
        target_mlp_module = module
        break
if target_mlp_name is None:
    raise ValueError("未能找到投影层(merger.mlp)，请检查模型结构！")

print(f"\n[手术前] 成功雷达锁定投影层路径: {target_mlp_name}")

# 动态获取原始输入输出维度
in_dim = target_mlp_module[0].in_features
# 反向遍历寻找最后一个out_features
out_dim = None
for layer in reversed(target_mlp_module):
    if hasattr(layer, 'out_features'):
        out_dim = layer.out_features
        break

print(f"\n[手术前] 探测到原始投影层维度 - 输入: {in_dim}, 输出: {out_dim}")

bottleneck_layer = BottleneckProjector(
    in_features=in_dim,
    out_features=out_dim,
    reduction_ratio=4
).to(device=model.device, dtype=torch.bfloat16)


# 动态替换模块 (黑科技核心)
# 如果路径是 "visual.merger.mlp"，找到 "visual.merger" 对象，然后替换它的 "mlp" 属性
parent_name = ".".join(target_mlp_name.split(".")[:-1])  # 获取父模块路径
child_name = target_mlp_name.split(".")[-1]              # 获取子模块名称

modules_dict = dict(model.named_modules())
parent_module = modules_dict[parent_name]

# 使用 setattr 进行替换
setattr(parent_module, child_name, bottleneck_layer)

print(f"[手术后] 成功将 {target_mlp_name} 替换为 Bottleneck 架构!\n")

# ================= 4. 测试“换心”后的模型是否能正常工作 (包含图像) =================
print("正在构造包含【图像+文本】的测试用例...")
# 创建一张纯黑的测试图像（仅用于跑通流程，尺寸满足 vit 要求即可）
from PIL import Image
test_image = Image.new('RGB', (224, 224), color = 'black')

messages =[
    {
        "role": "user",
        "content":[
            {"type": "image", "image": test_image},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

try:
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
except Exception as e:
    # 兼容性备用方案
    inputs = processor(
        text=[text],
        images=test_image,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

print("正在执行图像多模态前向传播测试...")
with torch.no_grad():
    outputs = model.forward(**inputs)

print("✅ 大功告成！带有 Bottleneck 架构的模型成功完成了多模态前向推理。")
print(f"新生成的特征能够顺利接入 LLM。")