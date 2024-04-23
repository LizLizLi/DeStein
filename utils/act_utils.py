import torch
from baukit import TraceDict
import numpy as np
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import sys
import pickle


def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print("The folder doesn't exist!")
        return -1
    files = os.listdir(folder_path)
    file_count = 0
    for file in files:
        if os.path.isfile(os.path.join(folder_path, file)):
            file_count += 1

    return file_count


def save_activations(activations):
    with open(
        "/nfs-data/user30/Projects/adapter-gpt2/data/paraDetox/att/gpt2-large/analysis/activations.pkl",
        "wb",
    ) as f:
        pickle.dump(activations, f)


def get_directions(
    hidden_states: int = 0,
    num_layers: int = 0,
):
    hidden_states_all = []
    train_y_all = []
    train_n_all = []
    y_all = []
    n_all = []
    num_demonstration = len(hidden_states)
    for demonstration_id in range(num_demonstration):
        y = torch.from_numpy(hidden_states[demonstration_id][0].flatten())
        n = torch.from_numpy(hidden_states[demonstration_id][1].flatten())
        y_all.append(y)
        n_all.append(n)

    y_matrix = torch.stack(y_all).view(num_demonstration, num_layers, -1)
    n_matrix = torch.stack(n_all).view(num_demonstration, num_layers, -1)

    for layer in range(num_layers):
        y_layer_matrix = y_matrix[:, layer, :].view(num_demonstration, -1)
        n_layer_matrix = n_matrix[:, layer, :].view(num_demonstration, -1)

        h = y_layer_matrix - n_layer_matrix

        train_y_all.append(y_layer_matrix)
        train_n_all.append(n_layer_matrix)

        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all)
    train_y_all = torch.stack(train_y_all)
    train_n_all = torch.stack(train_n_all)

    return train_y_all, train_n_all, fit_data


def get_act(model, inputs):
    all_layer_wise_activations = []
    all_head_wise_activations = []
    all_mlp_wise_activations = []
    model_name = model.config.model_type
    if "gpt2" in model_name:
        print("gpt2")
        LAYERS = [
            f"transformer.h.{i}.layer_out"
            for i in range(model.config.num_hidden_layers)
        ]
        HEADS = [
            f"transformer.h.{i}.attn.head_out"
            for i in range(model.config.num_hidden_layers)
        ]
        MLPS = [f"transformer.h.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    elif "llama" in model_name:
        print("llama")
        LAYERS = [
            f"model.layers.{i}.layer_out" for i in range(model.config.num_hidden_layers)
        ]
        HEADS = [
            f"model.layers.{i}.self_attn.head_out"
            for i in range(model.config.num_hidden_layers)
        ]
        MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    elif "opt" in model_name:
        print("opt")
        LAYERS = [
            f"model.decoder.layers.{i}.layer_out"
            for i in range(model.config.num_hidden_layers)
        ]
        HEADS = [
            f"model.decoder.layers.{i}.self_attn.head_out"
            for i in range(model.config.num_hidden_layers)
        ]
        MLPS = [
            f"model.decoder.layers.{i}.mlp"
            for i in range(model.config.num_hidden_layers)
        ]
    elif "mpt" in model_name:
        print("mpt")
        LAYERS = [
            f"transformer.blocks.{i}.layer_out"
            for i in range(model.config.num_hidden_layers)
        ]
        HEADS = [
            f"transformer.blocks.{i}.attn.head_out"
            for i in range(model.config.num_hidden_layers)
        ]
        MLPS = [
            f"transformer.blocks.{i}.mlp" for i in range(model.config.num_hidden_layers)
        ]

    for example_id in range(len(inputs)):
        layer = []
        head = []
        mlp = []
        for style_id in range(len(inputs[example_id])):
            with TraceDict(model, LAYERS + HEADS + MLPS) as ret:
                output = model(
                    input_ids=torch.tensor(inputs[example_id][style_id]["input_ids"])
                    .unsqueeze(0)
                    .cuda(),
                    attention_mask=torch.tensor(
                        inputs[example_id][style_id]["attention_mask"]
                    )
                    .unsqueeze(0)
                    .cuda(),
                    output_hidden_states=True,
                )
            hidden_states = output.hidden_states
            hidden_states = torch.stack(hidden_states, dim=0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()

            layer_wise_hidden_states = [
                ret[layer].output.squeeze().detach().cpu() for layer in LAYERS
            ]
            layer_wise_hidden_states = (
                torch.stack(layer_wise_hidden_states, dim=0).squeeze().numpy()
            )

            head_wise_hidden_states = [
                ret[head].output.squeeze().detach().cpu() for head in HEADS
            ]
            head_wise_hidden_states = (
                torch.stack(head_wise_hidden_states, dim=0).squeeze().numpy()
            )

            mlp_wise_hidden_states = [
                ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS
            ]
            mlp_wise_hidden_states = (
                torch.stack(mlp_wise_hidden_states, dim=0).squeeze().numpy()
            )

            layer.append(
                np.expand_dims(layer_wise_hidden_states[:, -8:, :].mean(1), axis=0)
            )
            head.append(
                np.expand_dims(head_wise_hidden_states[:, -8:, :].mean(1), axis=0)
            )
            mlp.append(
                np.expand_dims(mlp_wise_hidden_states[:, -8:, :].mean(1), axis=0)
            )
        all_layer_wise_activations.append(tuple(layer))
        all_head_wise_activations.append(tuple(head))
        all_mlp_wise_activations.append(tuple(mlp))

    num_layers = model.config.num_hidden_layers
    train_layer_y_all, train_layer_n_all, all_layer_wise_directions = get_directions(
        hidden_states=all_layer_wise_activations,
        num_layers=num_layers,
    )

    train_head_y_all, train_head_n_all, all_head_wise_directions = get_directions(
        hidden_states=all_head_wise_activations,
        num_layers=num_layers,
    )

    train_mlp_y_all, train_mlp_n_all, all_mlp_wise_directions = get_directions(
        hidden_states=all_mlp_wise_activations,
        num_layers=num_layers,
    )

    return (
        train_layer_y_all,
        train_layer_n_all,
        all_layer_wise_directions,
        train_head_y_all,
        train_head_n_all,
        all_head_wise_directions,
        train_mlp_y_all,
        train_mlp_n_all,
        all_mlp_wise_directions,
    )
