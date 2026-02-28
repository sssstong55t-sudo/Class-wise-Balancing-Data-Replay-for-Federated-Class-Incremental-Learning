import torch
from Local_Training import TTSLoss  # 假设你的 TTSLoss 类定义在 Local_Training.py


def train_local(model, task_loader, replay_buffer, num_old_classes, epochs=2, device='cuda'):
    # 论文 Stage 2: 类别增量优化
    criterion = TTSLoss(tau_old=0.9, tau_new=1.1, w_old=1.2, w_new=0.8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)[cite: 212]

    model.train()
    for epoch in range(epochs):
        # 将当前任务数据与重放缓存结合 [cite: 93]
        # 注意：这里的 loader 应该是结合了当前数据和 buffer 的混合数据
        for images, labels in task_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)

            # 使用 TTSLoss 缓解新旧类别之间的不平衡 [cite: 66, 184]
            # num_old_classes 用于告诉 Loss 函数哪些是属于过去任务的 Logits
            loss = criterion(logits, labels, num_old_classes)

            loss.backward()
            optimizer.step()

    return model.state_dict()