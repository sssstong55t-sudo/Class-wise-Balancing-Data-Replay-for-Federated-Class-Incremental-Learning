import numpy as np


def server_global_sampling(client_features_dict, client_labels_dict, m_total_budget):
    """
    client_features_dict: {client_id: X_prime_k} [cite: 148]
    client_labels_dict: {client_id: labels_k} (用于实现 Class-wise Balancing) [cite: 11]
    m_total_budget: 总重放预算 M [cite: 91]
    """
    client_ids = list(client_features_dict.keys())

    # 1. 拼接全局加密矩阵和标签
    X_global_prime = np.concatenate([client_features_dict[cid] for cid in client_ids], axis=0)[cite: 153]
    y_global = np.concatenate([client_labels_dict[cid] for cid in client_ids], axis=0)

    # 2. 执行 SVD 获取左奇异向量 U' [cite: 154]
    U_prime, _, _ = np.linalg.svd(X_global_prime, full_matrices=False)

    # 3. 计算所有样本的 Leverage Scores [cite: 163]
    leverage_scores = np.sum(U_prime ** 2, axis=1)

    # 4. 类别均衡采样逻辑 [cite: 11, 51]
    unique_classes = np.unique(y_global)
    samples_per_class = m_total_budget // len(unique_classes)
    final_selected_indices = []

    for cls in unique_classes:
        cls_idx = np.where(y_global == cls)[0]
        cls_scores = leverage_scores[cls_idx]

        # 归一化该类别的采样概率 [cite: 168]
        p_cls = cls_scores / np.sum(cls_scores)

        # 在该类别内根据杠杆得分采样 [cite: 175]
        selected = np.random.choice(
            cls_idx,
            size=min(len(cls_idx), samples_per_class),
            replace=False,
            p=p_cls
        )
        final_selected_indices.extend(selected)

    # 5. 将全局索引映射回客户端本地索引 [cite: 159]
    dispatch_results = {cid: [] for cid in client_ids}
    client_sample_counts = [client_features_dict[cid].shape[0] for cid in client_ids]
    cumulative_counts = np.cumsum([0] + client_sample_counts)

    for g_idx in final_selected_indices:
        for i, cid in enumerate(client_ids):
            if cumulative_counts[i] <= g_idx < cumulative_counts[i + 1]:
                dispatch_results[cid].append(int(g_idx - cumulative_counts[i]))
                break

    return dispatch_results