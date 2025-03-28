import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

MODEL_NAME = "google/gemma-3-1b-it"
PRETRAINED_MODEL_NAME = "google/gemma-3-1b-pt"
DEVICE = "mps"
OUTPUT_FILE = "layer_ablation_results_gemma.json"


def load_model_and_tokenizer(model_name):
    print(f"Chargement du modèle '{model_name}' et du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
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

    return layers


def reset_module_to_pretrained(model, layer_path, pretrained_model):
    def get_module(model, path):
        parts = path.split('.')
        if len(parts) == 1:
            return getattr(model, parts[0])
        else:
            return get_module(getattr(model, parts[0]), '.'.join(parts[1:]))

    target_module = get_module(model, layer_path)
    source_module = get_module(pretrained_model, layer_path)

    original_weights = {name: param.data.clone() for name, param in target_module.named_parameters()}

    with torch.no_grad():
        for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):
            target_param.data.copy_(source_param.data)

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

    print(f"Module {layer_path} restauré à ses poids finetunés")
    return True


def generate_response(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def interactive_chat(model, pretrained_model, tokenizer, layers):
    modified_layers = {}
    while True:
        user_input = input("\nEntrez votre prompt: ")

        if user_input.lower() == 'exit':
            break

        elif user_input.lower() == 'layers':
            layer_selection = input("> ")

            for layer_path, original_weights in modified_layers.items():
                restore_module_weights(model, layer_path, original_weights)
            modified_layers.clear()

            if layer_selection.lower() == 'all':
                selected_layers = layers
            else:
                selected_indices = [int(i.strip()) for i in layer_selection.split(',')]
                selected_layers = [layers[i] for i in selected_indices if 0 <= i < len(layers)]

            for layer_path in selected_layers:
                original_weights = reset_module_to_pretrained(model, layer_path, pretrained_model)
                if original_weights is not None:
                    modified_layers[layer_path] = original_weights

        elif user_input.lower() == 'reset':
            for layer_path, original_weights in modified_layers.items():
                restore_module_weights(model, layer_path, original_weights)
            modified_layers.clear()

        else:

            response = generate_response(model, tokenizer, user_input)
            print(f"\nRéponse : {response}")


def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    pretrained_model, _ = load_model_and_tokenizer(PRETRAINED_MODEL_NAME)
    layers = identify_layers(model)

    interactive_chat(model, pretrained_model, tokenizer, layers)

    print("Libération de la mémoire...")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
