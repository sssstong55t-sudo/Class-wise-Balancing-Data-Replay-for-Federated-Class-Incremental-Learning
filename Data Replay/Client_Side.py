import numpy as np
from scipy.linalg import orth


def get_pseudo_features(model, dataloader, device):
    model.eval()
    features_list = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # 假设你的 ResNet-18 有一个提取倒数第二层特征的方法 [cite: 146, 148]
            feat = model.extract_features(inputs)
            features_list.append(feat.cpu().numpy())

    X = np.concatenate(features_list, axis=0)  # (n_samples, d_features)

    # ISVD 加密逻辑 [cite: 145, 148]
    # 生成随机正交矩阵 P (n x n) 和 Q (d x d)
    P = orth(np.random.randn(X.shape[0], X.shape[0]))
    Q = orth(np.random.randn(X.shape[1], X.shape[1]))

    X_prime = P @ X @ Q  # 加密后的特征 [cite: 148]
    return X_prime