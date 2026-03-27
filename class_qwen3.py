import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList
import pandas as pd
class EntropyAdaptiveHook:
    """
    支持动态 Alpha 调整的 Hook
    """
    def __init__(self, model, steering_vector, target_layers_idx):
        self.model = model
        self.steering_vector = steering_vector.to(model.device)
        self.target_layers_idx = target_layers_idx
        self.current_alpha = 0.0
        self.hooks = []
        self.trigger_count = 0

    def register(self, model):
        # 适配 Qwen2/3-VL
        layer = model.model.language_model.layers[self.target_layers_idx]
        
        self.handle = layer.register_forward_hook(self.hook_fn)
        print(f"✅ 动态熵驱动 Hook 已挂载至 Layer {self.target_layers_idx}")
        

    def hook_fn(self, module, input, output):
        # 闭环控制：如果当前 Alpha 为 0，说明熵极低，无需干预，原样返回
        if self.current_alpha == 0.0:
            return output
        
        self.trigger_count += 1
        hidden_states = output[0] if isinstance(output, tuple) else output

        # 动态注入
        vec = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)

        new_hidden_states = hidden_states + self.current_alpha * vec

        if isinstance(output, tuple):
            return (new_hidden_states, ) + output[1:]
        
        return new_hidden_states

    def set_alpha(self, alpha):
        self.current_alpha = alpha
    
    def remove(self):
        if self.handle:
            self.handle.remove()



class EntropyDrivenSteeringProcessor(LogitsProcessor):
    """
    挂载在 model.generate 中的探针，实时计算熵并闭环控制 Hook
    """
    def __init__(self, hook_manager,threshold, base_alpha=30.0):
        self.hook_manager = hook_manager
        self.threshold = threshold
        self.base_alpha = base_alpha

        self.entropy_trace = []
        self.is_trigger_trace = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores 维度：[batch_size, vocab_size]
        probs = torch.softmax(scores, dim=-1)
        log_probs = torch.log_softmax(scores, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        self.entropy_trace.append(entropy)

        # Phase B 的核心逻辑：如果只做 Phase A 测定，传进来的 hook_manager 为 None
        if self.hook_manager is not None:
            if entropy > self.threshold:
                self.hook_manager.set_alpha(self.base_alpha)
                self.is_trigger_trace.append(1)
            else:
                self.hook_manager.set_alpha(0.0)
                self.is_trigger_trace.append(0)
        
        return scores



@torch.inference_mode()
def adaptive_generate(model, tokenizer, input_ids, image_inputs, hook_manager, entropy_threshold=1.5):
    # 1. 预处理
    # 需使用Qwen3-VL 的处理器处理 image_inputs
    input_ids = input_ids.to(model.device)
    past_key_values = None
    generated_ids = input_ids

    logits, past_key_values = model(input_ids, past_key_values=past_key_values, return_dict=True).values()

    # 2. 循环生成 (Docoding)
    for _ in range(max_new_tokens):
        # 计算熵
        probs = F.softmax(logits[:, -1, :], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # 熵驱动动态调整 Alpha
        new_alpha = 1.0 if entropy.item() > entropy_threshold else 0.0
        hook_manager.set_alpha(new_alpha)

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        output = model(next_token, past_key_values=past_key_values)
        logits, past_key_values = outpit.logits, output.kv_cache
    
    return generated_ids


def save_experiment_results(results_list, base_name="steering_report"):
    """
    保存实验结果至 CSV。支持追加写入，且自动处理非标量数据。
    """
    # 建议使用固定的文件名以防止生成过多零碎文件，或者按照需求动态命名
    filename = f"{base_name}.csv"

    # 1. 数据清洗：确保所有的 Tensor 或复杂对象都被转换为 float/str
    def clean_row(row):
        new_row = {}
        for k, v in row.items():
            if isinstance(v, torch.Tensor):
                new_row[k] = v.item() if v.numel() == 1 else v.tolist()
            elif isinstance(v, (list, dict)):
                new_row[k] = str(v)  # list/dict 转为字符串存入 CSV
            else:
                new_row[k] = v
        return new_row

    cleaned_list = [clean_row(row) for row in results_list]
    
    # 2. 转换为 DataFrame
    df = pd.DataFrame(cleaned_list)
    
    # 3. 保存逻辑
    # 如果文件已存在，则追加（不写 header）；如果不存在，则写入（写 header）
    import os
    file_exists = os.path.isfile(filename)
    
    df.to_csv(filename, mode='a' if file_exists else 'w', index=False, header=not file_exists)
    
    print(f"\n✅ 实验数据已实时保存/追加至: {filename}")