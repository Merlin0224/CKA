import torch
from injection import ResidualInjector, ChannelSaliencyGate
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from modelscope import snapshot_download
from dataload import build_pope_dataset, extract_object_from_caption
import pandas as pd
import glob
import numpy as np
import csv
import time
from class_qwen3 import EntropyAdaptiveHook, EntropyDrivenSteeringProcessor, adaptive_generate

def run_adaptive_eval(
    model,
    processor,
    pope_dataset,
    hook_manager=None,
    threshold=1.5,
    base_alpha=30.0
):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0

    all_entropy_traces = []
    print(f"\n🚀 开始评测 (Threshold={threshold}, Base_Alpha={base_alpha})...")

    for i, item in enumerate(pope_dataset):
        image = item['image']
        label_exists = item['gt']
        prompt = item['prompt']

        messages =[{"role": "user", "content":[{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        entropy_processor = EntropyDrivenSteeringProcessor(hook_manager, threshold, base_alpha)
        logits_processor = LogitsProcessorList([entropy_processor])

        with torch.no_grad():
            input_ids_len = input['input_ids'].shape[1]
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                logits_processor=logits_processor # 注入探针
            )

            response_ids = output_ids[0][input_ids_len:]
            response = processor.decode(response_ids, skip_special_tokens=True).lower().strip()

        all_entropy_traces.append({
            "gt": label_exists,
            "response": response,
            "entropy_trace": entropy_processor.entropy_trace
        })

        model_said_yes = response.startwith("yes")
        if label_exists and model_said_yes: tp += 1
        elif not label_exists and not model_said_yes: tn += 1
        elif not label_exists and model_said_yes: fp += 1 # 幻觉 (FP)
        elif label_exits and not model_said_yes: fn += 1

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fp) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision)
    hallucination_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

    res_dict ={
        "accuracy": acc * 100, "hallucination_rate": hallucination_rate, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }
    return res_dict, all_entropy_traces

def run_dynamic_layer_wise_experiment(
    model, processor, train_set, test_set,
    swwep_layers, thresholds, base_alpha=30.0, num_samples=50
):
    """
    结合 Phase A 和 Phase B 的终极实验管道。
    自动扫描多个 Layer，并在每个 Layer 上测试不同的熵触发阈值。
    """
    results_list = []

    # ==========================================
    # Phase A: Baseline 熵轨迹刻画 (无干预)
    # ==========================================
    print("\n" + "="*60)
    print("🌟 Phase A: Baseline 熵轨迹刻画 (提取 OOD 样本的不确定性分布)")
    print("="*60)

    # 传入 hook_manager=None, 纯粹作为探针测定分布
    baseline_res, baseline_traces = run_adaptive_eval(model, test_set, hook_manager=None)
    print(f"✅ Baseline 幻觉率: {calculate_hallucination_rate(baseline_res)}")

    results_list.append({"Layer": "Baseline", "Threshold": "N/A", "Alpha": 0.0, **baseline_res})
    
    with open("phase_a_entropy_traces.json", "w", encoding="utf-8") as f:
        json.dump(baseline_traces, f, ensure_ascii=False, indent=2)
    print("💾 Baseline 熵轨迹已保存至 phase_a_entropy_traces.json")

    # ==========================================
    # Phase B: 跨层级的熵驱动自适应动态干预
    # ==========================================
    print("\n" + "="*60)
    print("🔥 Phase B: 跨层级视觉向量提取 & 自适应阈值网格搜索")
    print("="*60)

    for layer_idx in sweep_layers:
        print(f"\n" + ">"*20 + f" 正在处理 Layer {layer_idx} " + "<"*20)

        # 1. 提取当前层的视觉忠实方向 (Steering Vector)
        # 注意：每次只用少量 train_set 样本提取，防止过拟合和节约时间
        steering_vec = extract_visual_truth_vector(model, processor, train_set, target_layer_idx=layer_idx, num_samples=num_samples)

        # 2. 挂载动态 Hook
        dynamic_hook = EntropyAdativeHook(steering_vector=steering_vector, target_layer_idx=layer_idx)
        dynamic_hook.register(model)

        # 3. 在当前层扫描不同的熵阈值
        for threshold in thresholds:
            print(f"\n--- [Layer {layer_idx}] 测试 Threshold: {threshold} | 基础注入强度 Alpha: {base_alpha} ---")

            res, _ = run_adaptive_eval(
                model, processor, test_set,
                hook_manager=dynamic_hook,
                threshold=threshold,
                base_alpha=base_alpha
            )

            result_str = calculate_hallucination_rate(res)
            print(f"📊 当前配置结果: {result_str}")

            # 记录数据
            results_list.append({
                "Layer": layer_idx,
                "Threshold": threshold,
                "Alpha": base_alpha,
                **res
            })

            save_experiment_results(results_list, base_name="dynamic_steering_report")


        # 4. 【SysML 核心防坑】：进入下一层前，必须彻底卸载当前层的 Hook
        dynamic_hook.remove()
        print(f"🧹 Layer {layer_idx} 的 Hook 已安全卸载。")
    
    return results_list


def extract_visual_truth_vector(model, processor, dataset, target_layer_idx, num_samples=20):
    print(f"\n🔍 正在 Layer {target_layer_idx} 提取视觉忠实方向 (Steering Vector)...")
    model.eval()

    activation_cache = {}
    def cache_hook(module, input, output):
        # 兼容不同的输出结构
        feat = output[0].detach() if isinstance(output, tuple) else output.detach()
        activation_cache['feat'] = feat

    layer_module = model.model.layers[target_layer_idx]
    handle = layer_module.register_forward_hook(cache_hook)

    pos_activations = []
    neg_activations = []
    for i, item in enumerate(dataset):
        if i >= num_samples: break
        image = item['image']

        # 构造输入
        messages =[{"role": "user", "content":[{"type": "image", "image": image}, {"type": "text", "text": item['prompt']}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        with torch,no_grad():
            model(**inputs)
            # 取最后一个文本 token (即生成的起手势位置)的激活状态
            last_token_feat = activation_cache['feat'][:, -1:, :].mean(dim=0, keepdim=True)

            if item['gt'] == True:
                pos_activations.append(last_token_feat)
            else:
                neg_activations.append(last_token_feat)
    
    handle.remove()

    # 计算差异向量
    pos_mean = torch.cat(pos_activations, dim=0).mean(dim=0, keepdim=True)
    neg_mean = torch.cat(neg_activations, dim=0).mean(dim=0, keepdim=True)
    steering_vector = pos_mean - neg_mean

    # 1. 计算 L2 Norm
    norm_val = torch.norm(steering_vector, p=2, dim=-1, keepdim=True)
    print(f"  -> [分析] 原始特征差异幅度 (Norm): {norm_val.item():.4f}")

    # 2. 向量归一化 (拉伸至单位球体)
    steering_vector = steering_vector / (norm_val + 1e-6)

    # 计算原本正样本向量的平均尺度，作为提示信息，
    # 这能帮助我们设定合理的 base_alpha 初始值。
    # 比如原始特征 norm 是 45，我们设 alpha=30 就是个合理的同一数量级。
    pos_norm_avg = torch.norm(pos_mean, p=2, dim=-1).item()
    print(f"  -> [提示] 建议的 Base Alpha 搜索上限不应超过其特征本身的尺度: {pos_norm_avg:.2f}")
    
    return steering_vector





def main():
    # ========== 1. 加载模型基建  ==========
    model_dir = "/root/autodl-tmp/.cache/modelscope/hub/models/qwen/Qwen3-VL-4B-Instruct/"
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, 
        device_map="auto", quantization_config=quantization_config, local_files_only=True
    )
    
    num_samples
    LOCAL_DATA_DIR = "/root/autodl-tmp/modelscope_datasets/flickr30k/data"
    test_set = build_pope_dataset(LOCAL_DATA_DIR, num_samples=num_samples)
    train_set = build_pope_dataset(LOCAL_DATA_DIR, num_samples=num_samples)

    sweep_layers = [4, 8, 12, 16, 20, 24]

    # thresholds 设定：
    # 1.0 (激进，稍微不确定就干预)
    # 1.5 (均衡)
    # 2.0 (保守，只有极度不确定、模型即将编造幻觉时才干预)
    thresholds =[1.0, 1.5, 2.0]
    base_alpha = [10.0, 20.0, 30.0]

    for alpha in base_alpha:
        all_results = run_dynamic_layer_wise_experiment(
            model=model,
            processor=processor,
            train_set=train_set,
            test_set=test_set,
            sweep_layers=sweep_layers,
            thresholds=thresholds,
            base_alpha=alpha
        )
    
    print("\n🎉 所有实验已完成！请查看生成的 CSV 报告。")

if __name__ == "__main__":
    main()