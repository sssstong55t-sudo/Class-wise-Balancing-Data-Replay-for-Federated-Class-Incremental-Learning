import torch
import torch.optim as optim
import numpy as np

# 从你上传的文件中导入必要的类和函数
from Data_Preparation import FCILDataPartition
from models import ResNet18Custom, aggregate, get_combined_loader
from Local_Training import TTSLoss
from Server_Side import server_global_sampling
from Client_Side import get_pseudo_features


def train_local_standard(model, loader, device):
    """阶段 1：初始任务使用标准交叉熵损失"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model.resnet.state_dict()  # 返回 state_dict 用于聚合


def train_local_tts(model, loader, old_classes_num, device):
    """阶段 2：增量任务使用 Task-aware Temperature Scaling (TTS) Loss"""
    # 参数根据论文设定：tau_old=0.9, tau_new=1.1
    criterion = TTSLoss(tau_old=0.9, tau_new=1.1, w_old=1.1, w_new=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels, old_classes_num)
        loss.backward()
        optimizer.step()
    return model.resnet.state_dict()


def federated_class_incremental_learning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clients = 5
    num_tasks = 5
    m_budget = 1000 if num_tasks == 5 else 500

    # 1. 数据准备
    partitioner = FCILDataPartition(num_clients=num_clients)
    tasks = partitioner.get_task_split(num_tasks=num_tasks)

    # 2. 初始化模型 (初始任务类别数，假设 CIFAR-100 平均分则为 20)
    initial_classes = len(tasks[0])
    global_model = ResNet18Custom(num_classes=initial_classes).to(device)

    client_buffers = {i: [] for i in range(num_clients)}
    total_classes_seen = 0

    for s in range(num_tasks):
        current_task_classes = tasks[s]
        print(f"\n--- Starting Task  {s + 1} (Category: {current_task_classes}) ---")

        # 如果不是第一个任务，扩展分类器
        if s > 0:
            new_classes_count = len(current_task_classes)
            global_model.expand_classifier(new_classes_count)
            global_model.to(device)

        # 获取当前任务的 Non-IID 数据索引
        client_data_indices = partitioner.partition_data_non_iid(current_task_classes)

        # 联邦训练轮次
        for r in range(100):  # 100 轮
            local_weights = []
            for k in range(num_clients):
                # 获取混合了 buffer 的 DataLoader
                current_client_data = torch.utils.data.Subset(partitioner.train_dataset, client_data_indices[k])
                train_loader = get_combined_loader(k, current_client_data, client_buffers[k])

                if s == 0:
                    w = train_local_standard(global_model, train_loader, device)
                else:
                    w = train_local_tts(global_model, train_loader, total_classes_seen, device)
                local_weights.append(w)

            # 服务器聚合
            avg_weights = aggregate(local_weights)
            global_model.resnet.load_state_dict(avg_weights)
            if r % 20 == 0:
                print(f"Round {r} completed.")

        # 更新已见类别总数
        total_classes_seen += len(current_task_classes)

        # 3. 任务结束时的 GDR 全局采样
        all_X_prime = {}
        all_Y = {}  # 用于存储标签

        for k in range(num_clients):
            subset = torch.utils.data.Subset(partitioner.train_dataset, client_data_indices[k])
            loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

            # 客户端返回特征和标签
            feat_prime, labels = get_pseudo_features(global_model, loader, device)
            all_X_prime[k] = feat_prime
            all_Y[k] = labels

        # 调用新的类别均衡采样函数
        dispatch_indices_dict = server_global_sampling(all_X_prime, all_Y, m_budget)

        # 4. 客户端根据指示更新 Buffer
        for k in range(num_clients):
            selected_indices = dispatch_indices_dict[k]
            # 从当前数据中提取样本存入 buffer
            # 简化处理：存储 (tensor, label) 元组
            current_subset = torch.utils.data.Subset(partitioner.train_dataset, client_data_indices[k])
            for idx in selected_indices:
                img, label = current_subset[idx]
                client_buffers[k].append((img, label))

    print("\nAll incremental learning tasks have been completed.")


if __name__ == "__main__":
    federated_class_incremental_learning()