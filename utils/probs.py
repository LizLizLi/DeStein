from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    tensor:(batch,seq_length,1280)
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


def _merge_heads(tensor, num_heads, attn_head_size):
    """
    Merges attn_head_size dim and num_attn_heads dim into hidden_size
    tensor:(batch, head, seq_length, head_features)
    return:(batch, seq_length, 1280)
    """
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)


def get_top_heads(
    all_train_head_y,
    all_train_head_n,
    num_layers,
    num_heads,
    num_to_intervene,
    seed,
):
    second_dim_size = all_train_head_y.shape[1]
    eighty_percent_size = int(second_dim_size * 0.8)
    train_positive = all_train_head_y[:, :eighty_percent_size, :]
    test_positive = all_train_head_y[:, eighty_percent_size:, :]
    train_negative = all_train_head_n[:, :eighty_percent_size, :]
    test_negative = all_train_head_n[:, eighty_percent_size:, :]
    train_labels = torch.cat(
        (torch.ones(eighty_percent_size), torch.zeros(eighty_percent_size))
    )
    test_labels = torch.cat(
        (
            torch.ones(second_dim_size - eighty_percent_size),
            torch.zeros(second_dim_size - eighty_percent_size),
        )
    )

    all_layer_head_accs = []
    probes = []
    for layer in range(num_layers):

        train_data = torch.cat((train_positive[layer], train_negative[layer]), dim=0)
        test_data = torch.cat((test_positive[layer], test_negative[layer]), dim=0)

        train_data = torch.unsqueeze(train_data, dim=1)
        test_data = torch.unsqueeze(test_data, dim=1)

        train_data = _split_heads(
            train_data.float(),
            num_heads,
            int(train_data.size(2) / num_heads),
        ).permute(0, 2, 1, 3)
        test_data = _split_heads(
            test_data.float(),
            num_heads,
            int(test_data.size(2) / num_heads),
        ).permute(0, 2, 1, 3)

        for head in range(num_heads):
            train_data_head = train_data[:, :, head, :].squeeze(dim=2).squeeze(dim=1)
            test_data_head = test_data[:, :, head, :].squeeze(dim=2).squeeze(dim=1)

            shuffle_indices = torch.randperm(train_data_head.size(0))
            train_data_head_shuffled = train_data_head[shuffle_indices]
            train_labels_shuffled = train_labels[shuffle_indices]

            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(
                train_data_head_shuffled.numpy(), train_labels_shuffled.numpy()
            )
            label_pred = clf.predict(test_data_head)

            all_layer_head_accs.append(accuracy_score(test_labels, label_pred))
            probes.append(clf)
    print("all_layer_head_accs:", all_layer_head_accs)

    all_layer_head_accs = np.array(all_layer_head_accs).reshape(num_layers, num_heads)
    top_heads = []
    top_accs = np.argsort(all_layer_head_accs.reshape(num_heads * num_layers))[::-1][
        :num_to_intervene
    ]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    matrix = torch.zeros(num_layers, num_heads, train_data.size(3))  # [36,20,64]

    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if (i, j) in top_heads:
                matrix[i, j, :] = 1

    matrix = _merge_heads(
        matrix.unsqueeze(dim=1).permute(0, 2, 1, 3),
        num_heads,
        train_data.size(3),
    )
    return matrix, top_heads, all_layer_head_accs


def train_probes_head_wise(
    seed,
    train_set_idxs,
    val_set_idxs,
    separated_head_wise_activations,
    separated_labels,
    num_layers,
    num_heads,
):
    all_head_accs = []
    probes = []

    all_X_train = np.concatenate(
        [separated_head_wise_activations[i] for i in train_set_idxs], axis=0
    )
    all_X_val = np.concatenate(
        [separated_head_wise_activations[i] for i in val_set_idxs], axis=0
    )
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis=0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis=0)

    for layer in tqdm(range(num_layers)):
        for head in range(num_heads):
            X_train = all_X_train[:, layer, head, :]
            X_val = all_X_val[:, layer, head, :]

            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(
                X_train, y_train
            )
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np


def train_probes_layer_wise(
    seed,
    train_set_idxs,
    val_set_idxs,
    layer_wise_activations,
    labels,
    num_layers,
):
    all_layer_accs = []
    probes = []

    all_X_train = np.concatenate(
        [layer_wise_activations[i] for i in train_set_idxs], axis=0
    )
    all_X_val = np.concatenate(
        [layer_wise_activations[i] for i in val_set_idxs], axis=0
    )

    y_train = np.concatenate([labels[i] for i in train_set_idxs], axis=0)
    y_val = np.concatenate([labels[i] for i in val_set_idxs], axis=0)

    for layer in tqdm(range(num_layers)):
        X_train = all_X_train[:, layer, :]
        X_val = all_X_val[:, layer, :]

        clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        all_layer_accs.append(accuracy_score(y_val, y_val_pred))
        probes.append(clf)

    all_layer_accs_np = np.array(all_layer_accs)

    return probes, all_layer_accs_np
