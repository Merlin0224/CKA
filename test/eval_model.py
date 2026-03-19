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
import csv
import time

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


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def save_experiment_results(results_list, base_name="steering_report"):
    timestamp = get_timestamp()
    filename = f"{base_name}_{timestamp}.csv"

    df = pd.DataFrame(results_list)
    df.to_csv(filename, index=False)
    print(f"\n✅ 实验数据已实时保存至: {filename}")

# =============== 激活干预模块 ===============
class ActivationSteeringHook:
    def __init__(self, steering_vector=None, alpha=1.0):
        """
        steering_vector: 提取出的“视觉忠实方向”向量 [1, 1, hidden_dim]
        alpha: 干预强度
        """
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.trigger_count = 0 
    
    def __call__(self, module, input, output):
        # 只有在推理生成阶段，并且有设定干预向量时才执行注入
        if self.steering_vector is not None:
            if self.trigger_count == 0:
                print(f"  [Hook Debug] 激活干预已成功触发! 强度 Alpha = {self.alpha}")
            self.trigger_count += 1

            hidden_states = output[0] if isinstance(output, tuple) else output

            # 维度对齐：确保 steering_vector 在当前设备上
            vec = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)

            # 强行在隐空间中叠加"视觉忠实方向"
            new_hidden_states = hidden_states + self.alpha * vec

            if isinstance(output, tuple):
                return (new_hidden_states, ) + output[1:]
            return new_hidden_states
        return output


# ============== 计算"视觉忠实方向" ===============
def extract_visual_truth_vector(model, processor, dataset, target_layer_idx, num_samples=20):
    """
    通过对比“图文匹配(正样本)”和“图文不匹配(负样本)”的内部激活差异，
    提取出代表“关注视觉事实”的几何方向向量。
    """
    print(f"\n🔍 正在 Layer {target_layer_idx} 提取视觉忠实方向 (Steering Vector)...")
    model.eval()

    # 临时探针: 用于抓取指定层的隐藏状态
    activation_cache = {}
    def cache_hook(module, input, output):
        activation_cache['feat'] = output[0].detach() if isinstance(output, tuple) else output.detach()
    
    layer_module = model.model.language_model.layers[target_layer_idx]
    handle = layer_module.register_forward_hook(cache_hook)

    pos_activations = []
    neg_activations = []

    for i, item in enumerate(dataset):
        if i >= num_samples: break
        image = item['image']

        # 正样本前向传播(GT=True) -> 获取关注图像时的状态
        if item['gt'] == True:
            messages =[{"role": "user", "content":[{"type": "image", "image": image}, {"type": "text", "text": item['prompt']}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            # 取最后一个 token 的激活值作为句子表征
            pos_activations.append(activation_cache['feat'][:, -1:, :].mean(dim=0, keepdim=True))
        
        # 负样本前向传播(GT=False) -> 获取产生幻觉冲动时的状态
        else:
            messages =[{"role": "user", "content":[{"type": "image", "image": image}, {"type": "text", "text": item['prompt']}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
            neg_activations.append(activation_cache['feat'][:, -1:, :].mean(dim=0, keepdim=True))

    handle.remove() # 卸载探针
    
    # 计算均值差异向量(Mean Difference Vector)
    # 这个向量的物理意义：从"瞎编"的隐空间指向"忠实于图像"的隐空间
    pos_mean = torch.cat(pos_activations, dim=0).mean(dim=0, keepdim=True)
    neg_mean = torch.cat(neg_activations, dim=0).mean(dim=0, keepdim=True)

    steering_vector = pos_mean - neg_mean

    norm_val = torch.norm(steering_vector, p=2, dim=-1, keepdim=True)
    print(f"  -> 原始向量幅度 (Norm): {norm_val.item():.6f}")
    
    # 强制将向量拉伸到长度为 1，这样 Alpha 的控制才有物理意义
    steering_vector = steering_vector / (norm_val + 1e-6) 
    return steering_vector

def run_layer_wise_steering_experiment(model, processor, dataset, test_dataset):
    # 要探查的广度: 扫描大模型浅、中、深各个层级
    sweep_layers = [2, 6, 10, 14, 18, 22]
    alphas = [10.0, 30.0] # 不同的干预强度

    # 记录实验结果的字典
    experiment_results = {}
    results_list = []
    # baseline(无任何干预)
    print("\n================ [Baseline] ================")
    baseline_res = run_eval(model, processor, test_dataset)
    experiment_results['Baseline'] = calculate_hallucination_rate(baseline_res)
    print(f"Baseline: {experiment_results['Baseline']}")
    results_list.append({"Layer": "Baseline", "Alpha": 0.0, **baseline_res})
    # 自动化进行大规模层级探查
    for layer_idx in sweep_layers:
        # 提取当前层的干预向量
        steering_vec = extract_visual_truth_vector(model, processor, dataset, layer_idx, num_samples=50)

        for alpha in alphas:
            print(f"\n================ [Layer {layer_idx} | Alpha {alpha}] ================")

            # 挂载激活干预 Hook
            steering_hook_obj = ActivationSteeringHook(steering_vector=steering_vec, alpha=alpha)
            layer_module = model.model.language_model.layers[layer_idx]
            handle = layer_module.register_forward_hook(steering_hook_obj)

            res = run_eval(model, processor, test_dataset)
            result_str = calculate_hallucination_rate(res)

            experiment_results[f'Layer_{layer_idx}_Alpha_{alpha}'] = result_str
            print(f"Result: {result_str}")
            row = {"Layer": layer_idx, "Alpha": alpha, **res}
            results_list.append(row)

            save_experiment_results(results_list)

            handle.remove()
    
    return experiment_results
    



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
    
    # model = load_model_and_injectors(model)
    LOCAL_DATA_DIR = "/root/autodl-tmp/modelscope_datasets/flickr30k/data"
    test_set = build_pope_dataset(LOCAL_DATA_DIR, num_samples=50)
    train_set = build_pope_dataset(LOCAL_DATA_DIR, num_samples=100)
    
    # # 2.实验 A: Baseline 测试(关闭注入)
    # set_injector_gate(model, 0.0)
    # baseline_results = run_eval(model, processor, test_set)

    # # 训练
    # set_injector_gate(model, 0.1)
    # train_injection_modules(model, processor, train_set, steps=30)
    # # 3.实验 B: Proposed 测试(开启注入)
    
    # proposed_results = run_eval(model, processor, test_set)

    # # 4.打印对比报告
    # print("\n🔥 [对比实验报告]")
    # print(f"Baseline 幻觉率: {calculate_hallucination_rate(baseline_results)}")
    # print(f"Proposed 幻觉率: {calculate_hallucination_rate(proposed_results)}")

    all_results = run_layer_wise_steering_experiment(model, processor, train_set, test_set)
    print(all_results)

if __name__ == "__main__":
    main()