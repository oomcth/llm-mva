import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "mps"


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
    else:
        for name, module in model.named_modules():
            if "DecoderLayer" in str(type(module)):
                layers.append(name)
    if not layers:
        print("Aucune couche de décodeur.")
    return layers


def remove_layers(model, layer_indices):
    modified_model = copy.deepcopy(model)

    parent_module = modified_model.model.layers

    num_layers = len(parent_module)
    invalid_indices = [idx for idx in layer_indices if idx < 0 or idx >= num_layers]

    sorted_indices = sorted(layer_indices, reverse=True)

    new_layers = [layer for i, layer in enumerate(parent_module) if i not in sorted_indices]

    modified_model.model.layers = torch.nn.ModuleList(new_layers)

    modified_model.config.num_hidden_layers = len(new_layers)

    return modified_model


def chat_with_model(model, tokenizer, max_new_tokens=100):
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == "exit":
            print("Fin de la conversation.")
            break

        inputs = tokenizer(user_input, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Modèle: {response}")


def main():

    model, tokenizer = load_model_and_tokenizer()

    user_input = input("Entrez les indices des couches à supprimer: ")

    if user_input.strip():
        layer_indices = [int(idx.strip()) for idx in user_input.split(",")]

        modified_model = remove_layers(model, layer_indices)
        if modified_model is None:
            modified_model = model
    else:
        modified_model = model

    modified_model = modified_model.to(DEVICE)

    chat_with_model(modified_model, tokenizer)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
