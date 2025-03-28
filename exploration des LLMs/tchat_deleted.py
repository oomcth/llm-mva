import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "mps"


def load_model_and_tokenizer():
    print("Chargement du modèle et du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(DEVICE)
    return model, tokenizer


def remove_layers(model, layer_indices):
    modified_model = copy.deepcopy(model)

    try:
        parent_module = modified_model.model.layers

        num_layers = len(parent_module)
        invalid_indices = [idx for idx in layer_indices if idx < 0 or idx >= num_layers]
        sorted_indices = sorted(layer_indices, reverse=True)

        layers = list(parent_module)
        for layer_idx in sorted_indices:
            layers.pop(layer_idx)
            print(f"Couche model.layers.{layer_idx} supprimée avec succès.")

        while len(parent_module) > 0:
            parent_module.pop(0)

        for layer in layers:
            parent_module.append(layer)

        modified_model.config.num_hidden_layers = len(parent_module)
        print(f"Nombre total de couches restantes: {len(parent_module)}")

        return modified_model
    except Exception as e:
        print(f"Erreur lors de la suppression des couches: {e}")
        return None


def chat_with_model(model, tokenizer, max_new_tokens=100):
    print("Entrez votre message.")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == "exit":
            print("Fin de la conversation.")
            break

        inputs = tokenizer(user_input, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
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


if __name__ == "__main__":
    main()
