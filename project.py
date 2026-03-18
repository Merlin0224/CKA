import torch
import torch.nn as nn

class BottleneckProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=3584, reduction_ratio=4):
        """
        input_dim: 视觉编码器(如 CLIP/SigLIP)输出的特征维度
        hidden_dim: LLM 的输入特征维度
        reduction_ratio: 瓶颈压缩比例
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // reduction_ratio),
            nn.LayerNorm(hidden_dim // reduction_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim // reduction_ratio, hidden_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        return self.net(x)


# 实例化一个模块进行测试
# Qwen2-VL 的视觉特征维度通常是 1152，LLM 输入是 3584
bottleneck = BottleneckProjector(input_dim=1152, hidden_dim=3584)
print("瓶颈投影层架构定义完毕：")
print(bottleneck)