import torch
from injection import ResidualInjector
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from modelscope import snapshot_download
import spacy
import pandas as pd
import pyarrow.parquet as pq
import random
import io
import os
from PIL import Image
import pyarrow.dataset as ds
import pandas as pd
import glob
import numpy as np

class VisualAligner(nn.Module):
    """
    一个轻量级的对齐模块，替代简单的 mean()
    """
    def __init__(self, visual_dim, hidden_dim):
        super().__init__()
        # 跨模态对齐：从原始视觉特征维度投影到 LLM 维度
        self.align_proj = nn.Linear(visual_dim, hidden_dim)

    
    def forward(self, visual_features):
        # 1. 映射到语言空间
        v = self.align_proj(visual_features)
        
        # 2. 跨序列求均值，提取全局语义 (Global Pool)
        # 无论视觉 Token 有多少个，最后都变成 1 个全局特征位点
        if v.dim() == 3: # [Batch, Seq, Dim]
            v_global = v.mean(dim=1, keepdim=True) # [Batch, 1, Dim]
        else: # [Seq, Dim]
            v_global = v.mean(dim=0, keepdim=True) # [1, Dim]
            
        return v_global
        
# 定义一个物理意义上的全局缓存，确保所有模块都能访问
GLOBAL_VLM_CACHE = {"visual_feat": None}

def set_injector_gate(model, value):
    """
    动态控制注入强度: 0.0 表示关闭注入(baseline), > 0 表示开启(proposed)
    """
    for name, module in model.named_modules():
        if isinstance(module, ResidualInjector):
            module.gate.data.fill_(value)
            print(f"✅ 注入强度 (Gate) 已设置为: {value}")
            

def run_eval(model, processor, pope_dataset):
    model.eval()
    # 初始化混淆矩阵计数器
    tp, tn, fp, fn = 0, 0, 0, 0
    
    print("\n🚀 正在进行 POPE 全指标评测...")

    for i, item in enumerate(pope_dataset):
        image = item['image']
        label_exists = item['gt']
        prompt = item['prompt']
        
        # 1. 推理过程 (确保只拿模型输出的第一个词)
        messages = [{"role": "user", "content":[{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            input_ids_len = inputs['input_ids'].shape[1]
            output_ids = model.generate(**inputs, max_new_tokens=5)
            response_ids = output_ids[0][input_ids_len:]
            response = processor.decode(response_ids, skip_special_tokens=True).lower().strip()
        
        # 2. 核心逻辑判断
        model_said_yes = response.startswith("yes")
        
        if label_exists and model_said_yes:
            tp += 1 # 正样本说对
        elif not label_exists and not model_said_yes:
            tn += 1 # 负样本说对
        elif not label_exists and model_said_yes:
            fp += 1 # 负样本说错 -> 幻觉！
        elif label_exists and not model_said_yes:
            fn += 1 # 正样本说错 -> 漏检！
        
        if (i+1) % 10 == 0:
            print(f"进度: [{i+1}/{len(pope_dataset)}]")

    # 3. 计算科学指标
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 幻觉率通常指 FP 在负样本中的比例
    hallucination_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    
    return {
        "accuracy": acc * 100,
        "hallucination_rate": hallucination_rate,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def build_pope_dataset(data_dir, num_samples=50):

    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    print(f"✅ 在目录 {data_dir} 中发现 {len(parquet_files)} 个数据分片。")
    
    dataset = ds.dataset(parquet_files, format="parquet")
    
    all_data = dataset.to_table().to_pandas()
    print(f"✅ 数据加载完毕，总条目数: {len(all_data)}")
    
    pope_data = []
    # 随机取 100 个下标用于后续的正负样本构造
    indices = random.sample(range(len(all_data)), num_samples * 2)
    
    for i in range(num_samples):
        # 取第 i 条作为正样本
        row_pos = all_data.iloc[indices[i]]
        img_bytes = row_pos['image']['bytes'] if isinstance(row_pos['image'], dict) else row_pos['image']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 提取名词
        caption_raw = row_pos['caption']
        if isinstance(caption_raw, (list, np.ndarray)):
            # 如果是列表/数组，取第一个元素，并确保转为 str
            caption = str(caption_raw[0])
        else:
            caption = str(caption_raw)
        obj_pos = extract_object_from_caption(caption)
        
        # 构造正样本
        pope_data.append({
            "image": image,
            "prompt": f"Is there a {obj_pos} in the image? Answer Yes or No.",
            "gt": True
        })
        
        # 构造负样本
        row_neg = all_data.iloc[indices[i + num_samples]]
        caption_neg = row_neg['caption'][0] if isinstance(row_neg['caption'], list) else row_neg['caption']
        obj_neg = extract_object_from_caption(caption_neg)
        
        if obj_neg != obj_pos:
            pope_data.append({
                "image": image,
                "prompt": f"Is there a {obj_neg} in the image? Answer Yes or No.",
                "gt": False
            })
            
    return pope_data


nlp = spacy.load("en_core_web_sm")
def extract_object_from_caption(caption):
    """
    通过 NLP 词性标注，自动提取 caption 中的名词 (NOUN)
    """
    # 强制类型转换：如果是数组/列表，取第一个元素，并强转为字符串
    if isinstance(caption, (list, np.ndarray)):
        caption_str = str(caption[0])
    else:
        caption_str = str(caption)
        
    doc = nlp(caption_str)
    
    # 提取 NOUN
    nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
    
    # 过滤掉无效词
    ignore_list = {'time', 'way', 'day', 'part', 'side', 'background', 'place', 'moment'}
    valid_nouns = [n for n in nouns if n not in ignore_list]
    
    return valid_nouns[0] if valid_nouns else "person"

def get_injection_hook(injector, aligner, visual_merger_module):
    # 传入 visual_merger_module 的引用
    def hook(module, input, output):
        hidden_states = output[0]
        visual_features = GLOBAL_VLM_CACHE.get("visual_feat")
        
        if visual_features is not None:
            # print("DEBUG: 正在执行残差注入...") # 调试用
            aligned_visual = aligner(visual_features)
            new_hidden_states = injector(hidden_states, aligned_visual)
            return (new_hidden_states,) + output[1:]
        else:
            # 如果走到这里，说明视觉特征没抓到
            print("DEBUG: 警告 - 未检测到视觉特征，跳过注入") 
            return output
        return output
    return hook

def visual_capture_hook(module, input, output):
    # 直接将 feat 存入当前 module 对象中
    # module 就是被挂载的 visual.merger
    feat = output[0] if isinstance(output, tuple) else output
    GLOBAL_VLM_CACHE["visual_feat"] = feat
    # print(f"DEBUG: 视觉特征已捕获，维度: {feat.shape}") # 调试用

def load_model_and_injectors(model):
    # 1. 挂载残差注入模块
    visual_dim = 3584
    hidden_dim = 3584
    
    injector = ResidualInjector(visual_dim, hidden_dim).to(device=model.device, dtype=torch.bfloat16)
    aligner = VisualAligner(visual_dim, hidden_dim).to(device=model.device, dtype=torch.bfloat16)

    model.add_module("residual_injector", injector)
    model.add_module("visual_aligner", aligner)

    visual_merger = None
    for name, module in model.named_modules():
        if "visual.merger" in name:
            visual_merger = module
            module.register_forward_hook(visual_capture_hook)
            break

    # 挂载注入 hook，传入这个 visual_merger 引用
    model.model.language_model.layers[6].register_forward_hook(
        get_injection_hook(injector, aligner, visual_merger)
    )
    print("✅ 残差注入模块已挂载至第 6 层！")
    return model

def calculate_hallucination_rate(results):
    """
    科研级结果解析器：从结果字典中提取核心指标并格式化。
    """
    if isinstance(results, (float, int)):
        return f"{results:.2f}%"

    if isinstance(results, dict):
        # 核心指标：幻觉率 (False Positive Rate)
        hr = results.get("hallucination_rate", 0)
        # 辅助指标：准确率和 F1，证明改进不是以牺牲识别能力为代价的
        acc = results.get("accuracy", 0)
        f1 = results.get("f1", 0)
        
        # 混淆矩阵数据：用于深度分析
        tp, tn, fp, fn = results.get("tp"), results.get("tn"), results.get("fp"), results.get("fn")
        
        # 返回一个格式化的科学字符串
        return (f"{hr:.2f}% [详细指标: Acc {acc:.2f}%, F1 {f1:.3f}] "
                f"(TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn})")

    return "N/A"

def train_injection_modules(model, processor, dataset, steps=20):
    print("\n🚀 开始极简微调 (Training Injector)...")
    # 冻结模型，只解冻自己添加
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    for name, param in model.named_parameters():
        if "residual_injector" in name or "visual_aligner" in name:
                param.requires_grad = True
                trainable_params.append(param)
    

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
    model.train()


    for i, item in enumerate(dataset):
        if i >= steps: break

        image = item['image']
        prompt = item['prompt'] # "Is there a[obj]...?"
        
        # 将 Ground Truth (True/False) 转换为模型的标准回答
        answer = "Yes." if item['gt'] else "No."

        # 构造标准的 VQA 问答对进行微调
        message =[
            {"role": "user", "content":[{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content":[{"type": "text", "text": answer}]}
        ]

        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        inputs["labels"] = inputs["input_ids"].clone()

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        print(f"Train Step [{i+1}/{steps}] - Loss: {loss.item():.4f}")

    print("✅ 微调完成！")


def main():
    # 1.加载模型与注入模块
    # model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")
    model_dir = "/root/autodl-tmp/.cache/modelscope/hub/models/qwen/Qwen2-VL-7B-Instruct/"
    # processor = AutoProcessor.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(
        model_dir, 
        local_files_only=True 
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        quantization_config=quantization_config,
        local_files_only=True
    )
    
    model = load_model_and_injectors(model)
    LOCAL_DATA_DIR = "/root/autodl-tmp/modelscope_datasets/flickr30k/data"
    test_set = build_pope_dataset(LOCAL_DATA_DIR, num_samples=1000)
    train_set = build_pope_dataset(LOCAL_DATA_DIR, num_samples=2000)
    
    # 2.实验 A: Baseline 测试(关闭注入)
    set_injector_gate(model, 0.0)
    baseline_results = run_eval(model, processor, test_set)

    # 训练
    set_injector_gate(model, 0.1)
    train_injection_modules(model, processor, train_set, steps=30)
    # 3.实验 B: Proposed 测试(开启注入)
    
    proposed_results = run_eval(model, processor, test_set)

    # 4.打印对比报告
    print("\n🔥 [对比实验报告]")
    print(f"Baseline 幻觉率: {calculate_hallucination_rate(baseline_results)}")
    print(f"Proposed 幻觉率: {calculate_hallucination_rate(proposed_results)}")

if __name__ == "__main__":
    main()