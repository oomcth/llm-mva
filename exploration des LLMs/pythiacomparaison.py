import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import os
from transformers.modeling_outputs import BaseModelOutputWithPast


def gram_matrix(x):
    return x @ x.transpose(-1, -2)


def cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    G_X, G_Y = gram_matrix(X), gram_matrix(Y)
    return (G_X * G_Y).sum() / (torch.norm(G_X) * torch.norm(G_Y))


def get_activations(model, tokenizer, text):
    activations = {}
    hooks = []

    def hook_fn(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]

            if isinstance(output, BaseModelOutputWithPast):
                output = output.last_hidden_state
            activations[layer_name] = output.detach().cpu().numpy()
        return hook

    for name, layer in model.named_modules():
        if "layers." in name:
            hooks.append(layer.register_forward_hook(hook_fn(name)))

    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        _ = model(**tokens)

    for hook in hooks:
        hook.remove()

    return activations


def compare_weights(model_pretrained, model_finetuned):
    deltas = {}
    for (name1, param1), (name2, param2) in zip(model_pretrained.named_parameters(), model_finetuned.named_parameters()):
        if param1.shape == param2.shape:
            deltas[name1] = torch.norm(param2 - param1, p=2).item()
    return deltas


def analyze_attention_heads(model_pretrained, model_finetuned):
    attention_diffs = {}
    for i, (layer1, layer2) in enumerate(zip(model_pretrained.layers, model_finetuned.layers)):
        attn1, attn2 = layer1.attention, layer2.attention
        diff = torch.norm(attn2.dense.weight - attn1.dense.weight, p=2).item()
        attention_diffs[f'Layer {i}'] = diff
    return attention_diffs


def plot_and_save(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 5))
    layers = list(data.keys())
    values = list(data.values())
    plt.plot(layers, values, marker='o', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.xticks(rotation=45)
    plt.savefig(f"comparaisonpythia1b/{filename}.png")
    plt.close()


def main():
    os.makedirs("comparaisonpythia1b", exist_ok=True)
    model_name = "EleutherAI/pythia-1b"
    model_finetuned_name = "EleutherAI/pythia-1b-deduped"
    text = "The quick brown fox jumps over the lazy dog."

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_pretrained = AutoModel.from_pretrained(model_name)
    model_finetuned = AutoModel.from_pretrained(model_finetuned_name)

    activations_pretrained = get_activations(model_pretrained, tokenizer, text)
    activations_finetuned = get_activations(model_finetuned, tokenizer, text)

    common_layers = set(activations_pretrained.keys()) & set(activations_finetuned.keys())
    cka_results = {layer: cka(torch.tensor(activations_pretrained[layer]), torch.tensor(activations_finetuned[layer])).item() for layer in common_layers}
    weight_deltas = compare_weights(model_pretrained, model_finetuned)
    attention_deltas = analyze_attention_heads(model_pretrained, model_finetuned)

    plot_and_save(cka_results, "CKA Similarity Across Layers", "Layers", "CKA Similarity", "cka_similarity")
    plot_and_save(weight_deltas, "Weight Differences (L2 Norm)", "Layers", "L2 Norm Difference", "weight_differences")
    plot_and_save(attention_deltas, "Attention Head Differences", "Layers", "L2 Norm Difference", "attention_differences")


if __name__ == "__main__":
    main()
