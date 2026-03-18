import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        temperature: 温度超参数，用于控制对比学习对“困难负样本”的敏感度
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, visual_features, text_features):
        """
        visual_features: [batch_size, feature_dim] (图像经过瓶颈层后的特征聚类)
        text_features: [batch_size, feature_dim] (文本描述的 Embedding)
        """
        # 1.归一化
        v_norm = F.normalize(visual_features, p=2, dim=-1)
        t_norm = F.normalize(text_features, p=2, dim=-1)

        # 2.计算余弦相似度
        # 矩阵对角线是正样本（匹配的图文），非对角线是负样本
        logits = torch.matmul(v_norm, t_norm.transpose(0, 1)) / self.temperature

        # 3.构造对比学习的目标标签（对角线为1，即 [0, 1, 2, ... batch_size - 1]
        batch_size =visual_features.shape[0]
        labels = torch.arange(batch_size, device=visual_features.device)

        # 4.计算双向 Cross Entropy
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.transpose(0, 1), labels)

        return (loss_v2t + loss_t2v) / 2.0


print("正在测试 InfoNCE 损失函数计算...")
dummy_v = torch.randn(4, 3584)
dummy_t = torch.randn(4, 3584)
criterion = InfoNCELoss(temperature=0.1)
loss = criterion(dummy_v, dummy_t)
print(f"✅ InfoNCE Loss 计算成功: {loss.item():.4f}")