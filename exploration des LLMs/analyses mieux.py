import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import seaborn as sns
import os


torch.manual_seed(42)
np.random.seed(42)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


model_name_it = "google/gemma-3-1b-it"
model_name_pt = "google/gemma-3-1b-pt"

tokenizer_it = AutoTokenizer.from_pretrained(model_name_it)
model_it = AutoModelForCausalLM.from_pretrained(model_name_it)

tokenizer_pt = AutoTokenizer.from_pretrained(model_name_pt)
model_pt = AutoModelForCausalLM.from_pretrained(model_name_pt)


dataset = load_dataset("hellaswag", split="train[:100]")


def get_all_submodule_activations(model, tokenizer, texts, submodules_to_analyze=None, max_length=512, batch_size=8, device="cpu"):
    if submodules_to_analyze is None:
        submodules_to_analyze = [None]

    model = model.to(device)
    model.eval()
    num_layers = len(model.model.layers)
    all_activations = {submodule: {layer_idx: [] for layer_idx in range(num_layers)} for submodule in submodules_to_analyze}

    def get_submodule(layer, submodule_name):
        if submodule_name is None:
            return layer
        return getattr(layer, submodule_name, layer)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)

        batch_activations = {
            submodule: {layer_idx: [[] for _ in range(len(batch_texts))] for layer_idx in range(num_layers)}
            for submodule in submodules_to_analyze
        }

        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            for submodule_name in submodules_to_analyze:
                submodule = get_submodule(layer, submodule_name)
                for text_idx in range(len(batch_texts)):
                    def hook_fn(submodule_name=submodule_name, layer_idx=layer_idx, text_idx=text_idx):
                        def fn(module, input, output):
                            batch_activations[submodule_name][layer_idx][text_idx].append(output[0].detach().cpu().numpy())
                        return fn
                    hook = submodule.register_forward_hook(hook_fn())
                    hooks.append(hook)

        with torch.no_grad():
            model(**inputs)

        for hook in hooks:
            hook.remove()

        for submodule_name in submodules_to_analyze:
            for layer_idx in range(num_layers):
                all_activations[submodule_name][layer_idx].extend(batch_activations[submodule_name][layer_idx])

    return all_activations


def pca_on_activation_differences(activations_it, activations_pt, n_components=10):

    differences = [act_it - act_pt for act_it, act_pt in zip(activations_it, activations_pt)]

    flattened_differences = []
    for diff in differences:
        flattened_diff = diff.reshape(1, -1)
        flattened_differences.append(flattened_diff)

    flattened_differences = np.concatenate(flattened_differences, axis=0)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flattened_differences)

    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance


def compute_weight_distances(model_it, model_pt, layer_idx=None):
    distances = {
        "L2": [],
        "cosine": [],
        "L1": []
    }

    if layer_idx is None:
        for i, (layer_it, layer_pt) in enumerate(zip(model_it.model.layers, model_pt.model.layers)):
            for (name_it, param_it), (name_pt, param_pt) in zip(layer_it.named_parameters(), layer_pt.named_parameters()):
                param_it = param_it.detach().cpu().numpy().flatten()
                param_pt = param_pt.detach().cpu().numpy().flatten()

                l2_dist = np.linalg.norm(param_it - param_pt)
                distances["L2"].append((i, name_it, l2_dist))

                cos_dist = cosine(param_it, param_pt)
                distances["cosine"].append((i, name_it, cos_dist))

                l1_dist = np.sum(np.abs(param_it - param_pt))
                distances["L1"].append((i, name_it, l1_dist))
    else:
        layer_it = model_it.model.layers[layer_idx]
        layer_pt = model_pt.model.layers[layer_idx]
        for (name_it, param_it), (name_pt, param_pt) in zip(layer_it.named_parameters(), layer_pt.named_parameters()):
            param_it = param_it.detach().cpu().numpy().flatten()
            param_pt = param_pt.detach().cpu().numpy().flatten()

            l2_dist = np.linalg.norm(param_it - param_pt)
            distances["L2"].append((layer_idx, name_it, l2_dist))

            cos_dist = cosine(param_it, param_pt)
            distances["cosine"].append((layer_idx, name_it, cos_dist))

            l1_dist = np.sum(np.abs(param_it - param_pt))
            distances["L1"].append((layer_idx, name_it, l1_dist))

    return distances


def save_pca_plots(pca_result, explained_variance, title, filename_prefix):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(f"{filename_prefix}_scatter.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
    plt.title(f"{title} - Variance expliquée par composante")
    plt.xlabel("Composante principale")
    plt.ylabel("Variance expliquée")
    plt.grid(True)
    plt.savefig(f"{filename_prefix}_explained_variance.png")
    plt.close()


def save_distances_plot(distances, metric, filename_prefix):
    layers = [d[0] for d in distances[metric]]
    dist_values = [d[2] for d in distances[metric]]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=dist_values, y=[f"Layer {l}" for l in layers], hue=[f"Layer {l}" for l in layers], palette="viridis", legend=False)
    plt.title(f"Distance {metric} entre les poids des couches")
    plt.xlabel(f"Distance {metric}")
    plt.ylabel("Couche")
    plt.savefig(f"{filename_prefix}_{metric}.png")
    plt.close()


if __name__ == "__main__":
    output_dir = "gemma22b_pca"
    os.makedirs(output_dir, exist_ok=True)

    texts = [example["ctx"] for example in dataset]

    num_layers = len(model_it.model.layers)

    submodules_to_analyze = [None, 'mlp', 'self_attn']

    activations_it = get_all_submodule_activations(
        model_it, tokenizer_it, texts, submodules_to_analyze=submodules_to_analyze, device=device
    )

    activations_pt = get_all_submodule_activations(
        model_pt, tokenizer_pt, texts, submodules_to_analyze=submodules_to_analyze, device=device
    )

    for submodule_name in submodules_to_analyze:
        submodule_label = "couche entière" if submodule_name is None else f"sous-module {submodule_name}"

        for layer_idx in range(num_layers):

            activations_it_layer = [act[0] for act in activations_it[submodule_name][layer_idx]]
            activations_pt_layer = [act[0] for act in activations_pt[submodule_name][layer_idx]]

            pca_result, explained_variance = pca_on_activation_differences(activations_it_layer, activations_pt_layer)

            filename_prefix = os.path.join(output_dir, f"{'layer' if submodule_name is None else submodule_name}_layer_{layer_idx}")
            plot_title = f"PCA des différences d'activations - {submodule_label.capitalize()} - Couche {layer_idx}"
            save_pca_plots(pca_result, explained_variance, plot_title, filename_prefix)

    distances = compute_weight_distances(model_it, model_pt)

    for metric in distances.keys():
        filename_prefix = os.path.join(output_dir, "distances")
        save_distances_plot(distances, metric, filename_prefix)
