import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from modelscope import snapshot_download

model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")

print(f"模型路径已确认为: {model_dir}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bn_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
# 1. 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(model_dir)

# 2. 定义 Hook 字典存储特征
storage = {}

def get_hook(layer_name):
    def hook(model, input, output):
       # Qwen2 的 output 是一个 tuple，第 0 个元素是 hidden_states
        if isinstance(output, tuple):
            storage[layer_name] = output[0].detach().cpu()
        else:
            storage[layer_name] = output.detach().cpu()
    return hook

# 3. 挂载 Hook 到第 12 层 (总共 28 层左右，12 层是语义表征的关键层)

target_layer = 12
target_layer_name = f"model.language_model.layers.{target_layer}"
print(f"准备挂载的目标路径是: {target_layer_name}")

modules_dict = dict(model.named_modules())
if target_layer_name in modules_dict:
    target_module = modules_dict[target_layer_name]
    target_module.register_forward_hook(get_hook(target_layer_name))
    print("hook挂载成功")
else:
    print("找不到指定的层，请检查路径。")


# 4. 构造一个简单的推理请求来测试 Hook 是否被触发
print("正在执行推理以触发 Hook...")
messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(text=[text], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("正在执行前向传播...")
with torch.no_grad():
    model.forward(**inputs)

# 5. 检查是否捕获成功
if target_layer_name in storage:
    print(f"\n[大功告成] 成功截获大模型内部特征！")
    print(f"特征张量的维度为: {storage[target_layer_name].shape}")
    print(f"这代表:[Batch Size, 序列长度(Token数), 隐藏层维度]")
else:
    print("\nHook 未触发。")