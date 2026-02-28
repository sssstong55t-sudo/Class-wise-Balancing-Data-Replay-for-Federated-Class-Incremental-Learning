import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class FCILDataPartition:
    def __init__(self, dataset_name='CIFAR100', num_clients=5, seed=2023):
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)

        # 1. 加载原始数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        # 这里的数据下载路径建议保持一致
        self.train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        # 2. 核心修改：建立标签映射表
        # 生成 0-99 的随机顺序
        self.class_order = np.arange(100)
        np.random.shuffle(self.class_order)

        # 创建映射字典：{原始标签: 新的连续索引}
        # 比如：如果 class_order[0] 是 83，那么 map[83] = 0
        self.class_map = {raw_label: new_idx for new_idx, raw_label in enumerate(self.class_order)}

        # 3. 将数据集中的所有标签替换为映射后的标签
        self.train_dataset.targets = [self.class_map[t] for t in self.train_dataset.targets]
        self.test_dataset.targets = [self.class_map[t] for t in self.test_dataset.targets]

    def get_task_split(self, num_tasks=5):
        """将映射后的 0-99 标签拆分为连续的任务块"""
        # 现在的 class_order 已经逻辑上变成了 0, 1, 2... 99
        all_mapped_classes = np.arange(100)
        cls_per_task = 100 // num_tasks
        return [all_mapped_classes[i * cls_per_task: (i + 1) * cls_per_task] for i in range(num_tasks)]

    def partition_data_non_iid(self, task_classes, beta=0.5):
        """
        使用 Dirichlet 分布进行非独立同分布数据划分
        task_classes: 当前任务映射后的类别列表 (例如任务1是 [0,1,...19])
        beta: 异构性参数
        """
        client_data_indices = [[] for _ in range(self.num_clients)]
        all_targets = np.array(self.train_dataset.targets)

        for cls in task_classes:
            idx_cls = np.where(all_targets == cls)[0]
            np.random.shuffle(idx_cls)

            # Dirichlet 采样
            proportions = np.random.dirichlet([beta] * self.num_clients)
            proportions = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]

            split_idx = np.split(idx_cls, proportions)
            for k in range(self.num_clients):
                client_data_indices[k].extend(split_idx[k])

        return client_data_indices