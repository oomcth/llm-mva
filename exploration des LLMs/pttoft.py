from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm


def create_hybrid_model(model_pt, model_ft, n_layers_pt, num_layers):
    with torch.no_grad():
        hybrid_model = AutoModelForCausalLM.from_pretrained(model_pt_name)  # Base saine
        hybrid_model.to(device)

        for i in range(n_layers_pt):
            hybrid_model.model.layers[i] = model_pt.model.layers[i]

        for i in range(n_layers_pt, num_layers):
            hybrid_model.model.layers[i] = model_ft.model.layers[i]

    return hybrid_model


def analyze_probabilities(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.cpu()


def save_token_evolution(probabilities_all_layers, top_k_indices, tokenizer, output_dir):
    for i in range(len(top_k_indices[0])):
        token_idx = top_k_indices[0, i].item()

        token = tokenizer.decode([token_idx])

        token_safe = f"token_{token_idx}"

        token_repr = repr(token).strip("'")

        token_dir = os.path.join(output_dir, token_safe)
        os.makedirs(token_dir, exist_ok=True)

        with open(os.path.join(token_dir, "token_info.txt"), "w") as f:
            f.write(f"Token ID: {token_idx}\n")
            f.write(f"Token repr: {token_repr}\n")
            f.write(f"Token raw: {token}")

        token_probs = [layer_probs[0, token_idx] for layer_probs in probabilities_all_layers]
        token_log_probs = np.log(token_probs)

        np.save(os.path.join(token_dir, "log_probabilities.npy"), token_log_probs)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(token_log_probs)), token_log_probs)
        plt.xlabel("Number of PT Layers")
        plt.ylabel("Log Probability")
        plt.title(f"Evolution of Log Probability for Token ID: {token_idx} ({token_repr})")
        plt.grid(True)
        plt.savefig(os.path.join(token_dir, "log_probability_plot.png"))
        plt.close()


def plot_combined_evolution(probabilities_all_layers, top_k_indices, tokenizer, output_dir, num_layers):
    top_k_tokens = []
    top_k_reprs = []

    for i in range(len(top_k_indices[0])):
        token_idx = top_k_indices[0, i].item()
        try:
            token = tokenizer.decode([token_idx])
            top_k_tokens.append(token)
            token_repr = repr(token).strip("'")
            top_k_reprs.append(f"{token_repr} (ID:{token_idx})")
        except Exception as e:
            print(f"Erreur lors du décodage de {token_idx}: {e}")
            top_k_tokens.append(f"Unknown")
            top_k_reprs.append(f"Unknown_ID:{token_idx}")

    top_k_log_probs = []

    for i in range(len(top_k_indices[0])):
        token_idx = top_k_indices[0, i].item()
        token_log_probs = [np.log(layer_probs[0, token_idx]) for layer_probs in probabilities_all_layers]
        top_k_log_probs.append(token_log_probs)

    plt.figure(figsize=(15, 8))
    for i, log_probs in enumerate(top_k_log_probs):
        plt.plot(range(num_layers + 1), log_probs, label=f"{top_k_reprs[i]}")

    plt.xlabel("Number of PT Layers")
    plt.ylabel("Log Probability")
    plt.title("Combined Evolution of Top-40 Token Log Probabilities")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=4, fontsize='small')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(output_dir, "combined_log_probability_plot.png"))

    plt.savefig(os.path.join(output_dir, "combined_log_probability_plot_no_legend.png"), 
                bbox_inches='tight')

    with open(os.path.join(output_dir, "token_legend.txt"), "w") as f:
        f.write("Token Legend:\n")
        for i, (token_repr, token) in enumerate(zip(top_k_reprs, top_k_tokens)):
            token_idx = top_k_indices[0, i].item()
            f.write(f"{token_idx}: {token_repr}\n")

    plt.close()


model_pt_name = "google/gemma-3-1b-ft"
model_ft_name = "google/gemma-3-1b-it"
prompt = "Who are you?\n\n"
top_k = 40
output_dir = "pttoft gemma saut"

os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_pt_name)
model_pt = AutoModelForCausalLM.from_pretrained(model_pt_name)
model_ft = AutoModelForCausalLM.from_pretrained(model_ft_name)
device = "mps"  # ou "cuda" si vous avez un GPU, ou "cpu"
model_pt.to(device)
model_ft.to(device)

num_layers = len(model_pt.model.layers)

probabilities_ft = analyze_probabilities(model_ft, tokenizer, prompt)
top_k_values, top_k_indices = torch.topk(probabilities_ft, top_k)

probabilities_all_layers = [probabilities_ft]  # Liste pour stocker les probabilités de chaque configuration

for n_layers_pt in tqdm(range(1, num_layers + 1), desc="Processing Layers"):  # +1 pour inclure le modèle PT complet
    hybrid_model = create_hybrid_model(model_pt, model_ft, n_layers_pt, num_layers)
    probabilities = analyze_probabilities(hybrid_model, tokenizer, prompt)
    probabilities_all_layers.append(probabilities)


save_token_evolution(probabilities_all_layers, top_k_indices, tokenizer, output_dir)
plot_combined_evolution(probabilities_all_layers, top_k_indices, tokenizer, output_dir, num_layers)

print(f"Results saved to: {output_dir}")
