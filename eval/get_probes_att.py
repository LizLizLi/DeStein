import sys

sys.path.append("/nfs-data/user30/Projects/adapter-gpt2")
import pickle
import torch
from utils.probs import _merge_heads, _split_heads
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "__main__" == __name__:
    parent_path = (
        "/nfs-data/user30/Projects/adapter-gpt2/data/paraDetox/att/gpt2-large/v8"
    )
    head_path = parent_path + "/head-{count}.pkl".format(count=20)
    acc_path = parent_path + "/head_acc.pkl"
    all_head_wise_directions = pickle.load(
        open(
            head_path,
            "rb",
        )
    )  # torch.Size([36, 1280])
    # directions = _split_heads(all_head_wise_directions, num_heads=20, attn_head_size=64)
    acc = pickle.load(
        open(
            acc_path,
            "rb",
        )
    )
    # print(directions.shape)
    # 23层第6个head (还可以); 14层第7个head (不好)
    layer = 23
    head = 6
    y_all = []
    n_all = []
    activations = pickle.load(
        open(
            "/nfs-data/user30/Projects/adapter-gpt2/data/paraDetox/att/gpt2-large/analysis/activations.pkl",
            "rb",
        )
    )
    # 遍历list，每个元素是一个turple，turple的两个元素都是list
    num_demonstration = len(activations)
    for demonstration_id in range(num_demonstration):
        y = torch.from_numpy(activations[demonstration_id][0])[
            :, layer - 1, :
        ].unsqueeze(0)
        n = torch.from_numpy(activations[demonstration_id][1])[
            :, layer - 1, :
        ].unsqueeze(
            0
        )  # torch.Size([1, 1, 1280])
        y = (_split_heads(y, num_heads=20, attn_head_size=64).squeeze(0).squeeze(1))[
            head - 1, :
        ]  # torch.Size([1, 20, 1, 64])
        n = (
            _split_heads(n, num_heads=20, attn_head_size=64)
            .squeeze(0)
            .squeeze(1)[head - 1, :]
        )

        y_all.append(y)
        n_all.append(n)

    y_stacked_tensor = torch.stack(y_all)
    y_numpy_matrix = y_stacked_tensor.numpy()
    n_stacked_tensor = torch.stack(n_all)
    n_numpy_matrix = n_stacked_tensor.numpy()
    # 合并两个矩阵
    X = np.vstack([y_numpy_matrix, n_numpy_matrix])
    labels = np.array([0] * len(y_numpy_matrix) + [1] * len(n_numpy_matrix))

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=20, n_iter=5000, random_state=16)
    X_2d = tsne.fit_transform(X)

    # 绘制所有句子的2D表示
    markers = ["o", "x"]  # 自定义两个类别的图标
    colors = ["#C4D6A0", "#D9958F"]  # 自定义两个类别的颜色
    plt.figure(figsize=(8, 6))
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis")
    for i, marker in enumerate(markers):
        plt.scatter(
            X_2d[labels == i, 0],
            X_2d[labels == i, 1],
            marker=marker,
            label=f"Class {i}",
            color=colors[i],
            cmap="viridis",
        )
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # 自定义图例旁边的文字
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    circle = mlines.Line2D(
        [],
        [],
        color="#C4D6A0",
        marker="o",
        linestyle="None",
        markersize=5,
        label="Non-toxic Sentence",
    )
    cross = mlines.Line2D(
        [],
        [],
        color="#D9958F",
        marker="x",
        linestyle="None",
        markersize=5,
        label="Toxic Sentence",
    )
    plt.legend(handles=[circle, cross], loc="best")
    plt.show()
