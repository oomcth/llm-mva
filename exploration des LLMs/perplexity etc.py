from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import evaluate
import torch
import torch.nn.functional as F


def create_hybrid_model(model_pt, model_ft, n_layers_pt, num_layers):
    with torch.no_grad():
        hybrid_model = AutoModelForCausalLM.from_pretrained(model_pt_name)
        hybrid_model.to(device)

        for i in range(n_layers_pt):
            hybrid_model.model.layers[i].load_state_dict(model_pt.model.layers[i].state_dict())
            hybrid_model.model.layers[i] = model_pt.model.layers[i]

        for i in range(n_layers_pt, num_layers):
            hybrid_model.model.layers[i].load_state_dict(model_ft.model.layers[i].state_dict())

    return hybrid_model


def calculate_perplexity(model, tokenizer, text):
    return 0


def compute_entropy(model, tokenizer, eval_text):
    encodings = tokenizer(eval_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings["input_ids"]
    encodings = {key: value.to("mps") for key, value in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    probs = F.softmax(logits, dim=-1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean(dim=-1)  # (batch_size, seq_len)

    return entropy.squeeze().cpu().numpy()


model_pt_name = "google/gemma-3-1b-pt"
model_ft_name = "google/gemma-3-1b-it"
# model_pt_name = "EleutherAI/pythia-70m-deduped"
# model_ft_name = "EleutherAI/pythia-70m"
eval_text = "The black cat that walks around the garden often rests under the tree near the window, where it enjoys watching the birds flying in the blue sky, while listening to the soothing sounds of the leaves gently moving in the light breeze."
output_dir = "lastchance"

os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_pt_name)
model_pt = AutoModelForCausalLM.from_pretrained(model_pt_name)
model_ft = AutoModelForCausalLM.from_pretrained(model_ft_name)
device = "mps"
model_pt.to(device)
model_ft.to(device)

num_layers = len(model_ft.model.layers)

perplexities = []
entropies = []
num_pt_layers_list = []

for n_layers_pt in tqdm(range(0, num_layers + 1), desc="Processing Layers"):
    hybrid_model = create_hybrid_model(model_pt, model_ft, n_layers_pt, num_layers)

    entropy = compute_entropy(hybrid_model, tokenizer, eval_text)
    print(entropy)
    entropies.append(entropy)

    num_pt_layers_list.append(n_layers_pt)


print(entropies)
