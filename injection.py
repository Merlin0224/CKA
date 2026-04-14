import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig, AutoProcessor
from modelscope import snapshot_download

# ============ 旁路残差注入模块 ===============
class ResidualInjector(nn.Module):
    def __init__(self, visual_dim, hidden_dim):
        super().__init__()
        # 映射网路，将视觉特征对齐到 LLM 空间
        self.proj = nn.Linear(visual_dim, hidden_dim)
        # 可学习的门控参数，初始化为 0，保证初始微调时模型输出完全等同于基座
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, hidden_states, visual_features):
        # 核心逻辑：hidden_states + gate * (Visual_Projection)
        injected_feature = self.proj(visual_features)
        return hidden_states + self.gate * injected_feature

class VisualAligner(nn.Module):
    """
    一个轻量级的对齐模块，替代简单的 mean()
    """
    def __init__(self, visual_dim, hidden_dim):
        super().__init__()
        # 1. 跨模态对齐：从原始视觉特征维度投影到 LLM 维度
        self.align_proj = nn.Linear(visual_dim, hidden_dim)
        # 2. 序列长度对齐：使用 AdaptiveAvgPool1d，这能保证无论视觉 Token 有多少，
        # 都能被映射到固定的 Token 长度，或者保持序列空间的一致性
        self.pool = nn.AdaptiveAvgPool1d(2) # 先压缩为 2 个全局语义表征
    
    def forward(self, visual_features, target_seq_len):
        # visual_features: [batch, visual_seq_len, visual_dim]
        # 1.投影到 LLM 空间: [batch, visual_seq_len, hidden_dim]
        v = self.align_proj(visual_features)

        # 2.简单的对齐逻辑：广播到当前的语言 Token 长度
        # [batch, 1, hidden_dim] -> [batch, target_seq_len, hidden_dim]
        # 这一步保证了视觉信息能够在序列维度上与每一个语言 Token 进行残差交互
        v_pooled = v.mean(dim=1, keepdim=True)
        return v_pooled.expand(-1, target_seq_len, -1)

class ChannelSaliencyGate(nn.Module):
    """
    通道显著性门控
    通过对特征通道的加权，抑制背景噪声，放大语义显著通道
    """
    # def __init__(self, channels, reduction=8):
    #     super().__init__()
    #     self.avg_pool = nn.AdaptiveAvgPool1d(1)
    #     self.fc = nn.Sequential(
    #         nn.Linear(channels, channels // reduction, bias=False),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(channels // reduction, channels, bias=False),
    #         nn.Sigmoid()
    #     )
    
    # def forward(self, x):
    #      # 自动探测维度：如果输入是 2D [Batch, Channels]，转为 3D 处理
    #     is_2d = (x.dim() == 2)
    #     if is_2d:
    #         x = x.unsqueeze(1) # 变为 [Batch, 1, Channels]
            
    #     b, s, c = x.shape
    #     # 1. 空间池化: [Batch, Channels, 1]
    #     y = self.avg_pool(x.permute(0, 2, 1))
    #     # 2. 预测权重: [Batch, Channels, 1]
    #     y = self.fc(y.view(b, c)).view(b, c, 1)
    #     # 3. 加权: [Batch, Seq, Channels]
    #     out = x * y.permute(0, 2, 1)
        
    #     return out.squeeze(1) if is_2d else out

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature # 控制对比度的超参数，无需训练
        
    def forward(self, x):
        # x 维度: [batch, seq_len, channels]
        
        # 1. 自动探测：如果是平铺的 2D 特征，直接放行（因为无法算空间方差）
        is_2d = (x.dim() == 2)
        if is_2d:
            return x
            
        # 2. 计算每个通道在空间(seq_len)维度上的方差 (Variance)
        # 结果维度: [batch, 1, channels]
        channel_variance = torch.var(x, dim=1, keepdim=True)
        
        # 3. Min-Max 归一化，将其转化为 0~1 之间的门控权重
        # 减去最小值，除以极差
        v_min = channel_variance.min(dim=-1, keepdim=True)[0]
        v_max = channel_variance.max(dim=-1, keepdim=True)[0]
        
        # 得到天然的显著性权重 (方差越大，权重越接近 1；方差越小，权重越接近 0)
        gate_weight = (channel_variance - v_min) / (v_max - v_min + 1e-6)
        
        # 4. (可选) 增加锐化机制
        # 用 temperature 控制过滤的强度
        gate_weight = torch.pow(gate_weight, self.temperature)
        
        # 5. 动态重加权：物理过滤背景噪声！
        return x * gate_weight

def get_injection_hook(injector_module, aligned_visual):
    def hook(module, input, output):
        hidden_states = output[0] # [batch, seq, hidden]
        visual_features = model.visual_hidden_states
        # 使用更专业的对齐逻辑
        # 视觉特征序列长度对齐到当前 LLM 的序列长度
        aligned_visual = aligned_visual(visual_features, hidden_states.size(1))
        new_hidden_states = injector_module(hidden_states, aligned_visual)
        return (new_hidden_states, ) + output[1:]
    return hook

def visual_capture_hook(module, input, output):
    model.visual_hidden_states = output[0]

def main():
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

    visual_dim = 3584 
    hidden_dim = 3584 

    injector = ResidualInjector(visual_dim, hidden_dim).to(device=model.device, dtype=torch.bfloat16)
    aligner = VisualAligner(visual_dim, hidden_dim).to(device=model.device, dtype=torch.bfloat16)
    for name, module in model.named_modules():
        if "visual.merger" in name:
            module.register_forward_hook(visual_capture_hook)
            break

    model.model.language_model.layers[6].register_forward_hook(get_injection_hook(injector, aligner))
    print("✅ 残差注入模块已挂载至第 6 层！")
    

if __name__ == "__main__":
    main()