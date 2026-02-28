import torch
import torch.nn as nn
import torch.nn.functional as F


class TTSLoss(nn.Module):
    def __init__(self, tau_old=0.9, tau_new=1.1, w_old=1.2, w_new=0.8):
        super(TTSLoss, self).__init__()
        self.tau_old = tau_old  # 旧类温度：使分布更锐利
        self.tau_new = tau_new  # 新类温度：使分布更平滑
        self.w_old = w_old  # 旧类权重因子
        self.w_new = w_new  # 新类权重因子

    def forward(self, logits, targets, old_classes_num):
        """
        logits: 模型输出 (batch_size, total_classes)
        targets: 标签
        old_classes_num: 之前任务的总类别数
        """
        # 1. 拆分 Logits
        old_logits = logits[:, :old_classes_num] / self.tau_old
        new_logits = logits[:, old_classes_num:] / self.tau_new

        # 合并缩放后的 Logits
        scaled_logits = torch.cat([old_logits, new_logits], dim=1)

        # 2. 根据样本所属类别（新/旧）应用权重
        # 创建权重掩码
        is_old = (targets < old_classes_num).float()
        weights = is_old * self.w_old + (1 - is_old) * self.w_new

        # 3. 计算加权交叉熵
        ce_loss = F.cross_entropy(scaled_logits, targets, reduction='none')
        return (ce_loss * weights).mean()

# 在你的本地训练循环中使用：
# criterion = TTSLoss()
# loss = criterion(outputs, labels, num_old_classes)