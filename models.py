import torch
import torch.nn as nn
from torchvision import models
import copy


class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=20):
        super(ResNet18Custom, self).__init__()
        # 加载标准的 ResNet-18
        self.resnet = models.resnet18(weights=None)
        # 获取全连接层之前的特征维度 (通常是 512)
        self.feature_dim = self.resnet.fc.in_features
        # 初始分类器
        self.resnet.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        """用于 GDR 模块提取倒数第二层特征 """
        # 运行到 avgpool 层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        return torch.flatten(x, 1)

    def expand_classifier(self, new_classes_count):
        """动态增加分类器维度以支持增量任务 """
        old_classes_count = self.resnet.fc.out_features
        total_classes = old_classes_count + new_classes_count

        old_weights = self.resnet.fc.weight.data
        old_bias = self.resnet.fc.bias.data

        # 创建新的全连接层
        self.resnet.fc = nn.Linear(self.feature_dim, total_classes)

        # 保留旧类别的权重
        with torch.no_grad():
            self.resnet.fc.weight[:old_classes_count] = old_weights
            self.resnet.fc.bias[:old_classes_count] = old_bias


def aggregate(local_weights):
    """标准的 FedAvg 聚合逻辑 [cite: 1, 16]"""
    avg_weights = copy.deepcopy(local_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    return avg_weights


def get_combined_loader(client_id, current_task_data, replay_buffer, batch_size=128):
    """
    将当前任务的新数据与 Buffer 中的旧数据混合 [cite: 93]
    replay_buffer 格式: [(image_tensor, label), ...]
    """
    # 提取 buffer 中的图片和标签
    if len(replay_buffer) > 0:
        images, labels = zip(*replay_buffer)
        buffer_dataset = torch.utils.data.TensorDataset(torch.stack(images), torch.tensor(labels))
        # 合并数据集 [cite: 93]
        combined_dataset = torch.utils.data.ConcatDataset([current_task_data, buffer_dataset])
    else:
        combined_dataset = current_task_data

    return torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)