import os
import io
import torch
import glob
import numpy as np
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from modelscope import snapshot_download
import pyarrow.parquet as pq
from PIL import Image


def center_kernel(K):
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    return torch.matmul(torch.matmul(H, K), H)

def linear_cka(X, Y):
    K = torch.matmul(X, X.t())
    L = torch.matmul(Y, Y.t())
    K_c = center_kernel(K)
    L_c = center_kernel(L)
    hsic = torch.sum(K_c * L_c)
    norm_k = torch.sqrt(torch.sum(K_c * K_c))
    norm_l = torch.sqrt(torch.sum(L_c * L_c))
    return (hsic / (norm_k * norm_l)).item()

def main():

    LOCAL_DATA_DIR = "/root/autodl-tmp/modelscope_datasets/flickr30k/data"
    parquet_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.parquet"))
    print(f"✅ 找到数据包: {parquet_files}")

    samples_processed = 0
    MAX_SAMPLES = 50

    
    # model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")
    # processor = AutoProcessor.from_pretrained(model_dir)

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"
    # )

    # print("正在加载模型...")
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_dir,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     quantization_config=quantization_config
    # )

    model_dir = "/root/autodl-tmp/.cache/modelscope/hub/models/qwen/Qwen3-VL-8B-Instruct/"
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4"
    )    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        quantization_config=quantization_config, 
        local_files_only=True
    )

    probe_layers = [0, 4, 8, 12, 16, 20, 24]
    storage = {}

    def get_hook(name):
        def hook(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            storage[name] = tensor.detach().reshape(-1, tensor.shape[-1]).float()
        return hook

    modules_dict = dict(model.named_modules())
    
    for layer_idx in probe_layers:
        possible_names = [
            f"model.layers.{layer_idx}", 
            f"model.language_model.layers.{layer_idx}",
            f"language_model.model.layers.{layer_idx}"
        ]
        for name in possible_names:
            if name in modules_dict:
                modules_dict[name].register_forward_hook(get_hook(f"layer_{layer_idx}"))
                break
    print("✅ 模型加载与探针挂载完成。")

    

    print(f"开始在 {MAX_SAMPLES} 个样本上计算表征动力学...")
    cka_results = {l:[] for l in probe_layers if l != 0}
    for p_file in parquet_files:
        if samples_processed >= MAX_SAMPLES: break
    
        # 直接读取 Parquet 表格
        table = pq.read_table(p_file)
        # 按行迭代
        for i in range(len(table)):
            if samples_processed >= MAX_SAMPLES: break
        
            # 提取图像二进制数据
            # 注意：在 HF 的 Arrow 格式中，image 字段通常是一个 struct，包含 'bytes'
            img_dict = table['image'][i].as_py()
            image = Image.open(io.BytesIO(img_dict['bytes'])).convert("RGB")
        
            # 构造输入
            prompt_text = "Describe this image."
            messages = [{"role": "user", "content":[{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
            inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt").to(model.device)
        
            storage.clear()
            with torch.no_grad():
                model.forward(**inputs)
        
            # 计算 CKA
            base_layer = storage.get("layer_0")
            if base_layer is not None:
                for l in probe_layers:
                    if l == 0: continue
                    if f"layer_{l}" in storage:
                        cka_results[l].append(linear_cka(base_layer, storage[f"layer_{l}"]))
        
            samples_processed += 1
            print(f"[{samples_processed}/{MAX_SAMPLES}] 样本计算完成...")
    
    print("\n🔥 [统计学实验结果] 表征动力学分析:")
    for l in probe_layers:
        if l == 0: continue
        if cka_results[l]:
            print(f"Layer 00 -> Layer {l:02d} | CKA: {np.mean(cka_results[l]):.4f} (±{np.std(cka_results[l]):.4f})")

if __name__ == "__main__":
    main()