import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import torch.nn.functional as F

MODEL_NAME = "google/gemma-3-1b-it"
DEVICE = "mps"
OUTPUT_FILE = "layer_ablation_results_gemma.json"


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(DEVICE)
    return model, tokenizer


def identify_layers(model):
    layers = []
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        layers = [f"model.layers.{i}" for i in range(num_layers)]

    if not layers:
        for name, module in model.named_modules():
            if "DecoderLayer" in str(type(module)):
                layers.append(name)

    return {
        "decoder_layers": layers,
        "attention_layers": [],
        "mlp_layers": []
    }


def evaluate_memory(model, tokenizer, n_samples=20):

    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation[:100]")

    correct = 0
    for item in tqdm(dataset, desc="Évaluation mémoire"):
        question = item["question"]
        choices = item["mc1_targets"]["choices"]

        scores = []
        for choice in choices:
            input_text = f"Question: {question}\nRéponse: {choice}"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
                logits = outputs.logits

            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            target_log_probs = torch.gather(log_probs, 2, inputs['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)
            mask = inputs['attention_mask'][:, 1:].float()
            score = (target_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
            score = score.item()
            scores.append(score)

        predicted_idx = np.argmax(scores)
        correct_idx = item["mc1_targets"]["labels"].index(1)

        if predicted_idx == correct_idx:
            correct += 1

    accuracy = correct / len(dataset)
    return {"memory_accuracy": accuracy}


def evaluate_reasoning(model, tokenizer, n_samples=100):
    return {"reasoning_accuracy": 0}


def evaluate_common_sense(model, tokenizer, device="mps", n_samples=100):
    dataset = load_dataset("hellaswag", split=f"validation[:{n_samples}]")

    num_correct = 0
    num_total = 0

    for item in tqdm(dataset, desc="Évaluation sens commun"):
        context = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        losses = []
        for ending in endings:
            input_text = f"{context} {ending}"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
                logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_tokens = inputs['input_ids'][..., 1:].contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(
                flat_shift_logits, flat_shift_tokens, reduction='none'
            )
            shift_losses = shift_losses.view(inputs['input_ids'].size(0), -1)

            shift_mask = inputs['attention_mask'][..., 1:].contiguous()
            masked_shift_losses = shift_losses * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)

            losses.append(avg_loss.item())

        pred_idx = np.argmin(losses)
        if pred_idx == label:
            num_correct += 1
        num_total += 1

    accuracy = num_correct / num_total
    return {"common_sense_accuracy": accuracy}


def reset_module_to_pretrained(model, layer_path, pretrained_model_name="google/gemma-3-1b-pt"):
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float16
    )

    def get_module(model, path):
        parts = path.split('.')
        if len(parts) == 1:
            return getattr(model, parts[0])
        else:
            return get_module(getattr(model, parts[0]), '.'.join(parts[1:]))

    target_module = get_module(model, layer_path)
    source_module = get_module(base_model, layer_path)

    with torch.no_grad():
        for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):
            target_param.data.copy_(source_param.data)

    return True


def randomize_module(model, layer_path):
    def get_module(model, path):
        parts = path.split('.')
        if len(parts) == 1:
            return getattr(model, parts[0])
        else:
            return get_module(getattr(model, parts[0]), '.'.join(parts[1:]))

    module = get_module(model, layer_path)

    original_weights = {}
    for name, param in module.named_parameters():
        original_weights[name] = param.data.clone()

    with torch.no_grad():
        for name, param in module.named_parameters():
            std = param.data.std()
            param.data = torch.randn_like(param.data) * std * 0.1

    print(f"Module {layer_path} randomisé")
    return original_weights


def restore_module_weights(model, layer_path, original_weights):
    def get_module(model, path):
        parts = path.split('.')
        if len(parts) == 1:
            return getattr(model, parts[0])
        else:
            return get_module(getattr(model, parts[0]), '.'.join(parts[1:]))

    module = get_module(model, layer_path)

    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])

    return True


def run_layer_ablation_study(use_randomization=True):
    results = {}

    model, tokenizer = load_model_and_tokenizer()

    layer_groups = identify_layers(model)

    target_layers = layer_groups["decoder_layers"]

    print("Évaluation du modèle de base...")
    baseline_memory = evaluate_memory(model, tokenizer)
    baseline_reasoning = evaluate_reasoning(model, tokenizer)
    baseline_common_sense = evaluate_common_sense(model, tokenizer)

    baseline_results = {
        **baseline_memory,
        **baseline_reasoning,
        **baseline_common_sense
    }

    results["baseline"] = baseline_results
    print(f"Résultats de base: {baseline_results}")

    for layer_path in target_layers:

        modified_model = copy.deepcopy(model)
        modified_model = modified_model.to(DEVICE)

        if use_randomization:
            original_weights = randomize_module(modified_model, layer_path)
            method = "randomized"
        else:
            success = reset_module_to_pretrained(modified_model, layer_path)
            method = "reset_to_pretrained"

        memory_results = evaluate_memory(modified_model, tokenizer)
        reasoning_results = evaluate_reasoning(modified_model, tokenizer)
        common_sense_results = evaluate_common_sense(modified_model, tokenizer)

        layer_results = {
            **memory_results,
            **reasoning_results,
            **common_sense_results
        }

        results[f"{method}_{layer_path}"] = layer_results

        del modified_model
        torch.mps.empty_cache()

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def plot_results(results):
    metrics = ["memory_accuracy", "reasoning_accuracy", "common_sense_accuracy"]
    baseline = results["baseline"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        layer_names = [key for key in results.keys() if key != "baseline"]
        values = [results[key].get(metric, 0) for key in layer_names]
        baseline_value = baseline.get(metric, 0)

        plt.bar(range(len(layer_names)), values, label=f"{metric} modifié")
        plt.axhline(baseline_value, color='r', linestyle='--', label=f"Baseline {metric}")

        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.title(f"Impact de la modification des couches sur {metric}")
        plt.xlabel("Couche modifiée")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_ablation_gemma.png")
        plt.close()


if __name__ == "__main__":
    results = run_layer_ablation_study(use_randomization=False)
    plot_results(results)
