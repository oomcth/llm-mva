import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEVICE = "mps"
PRETRAINED_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


def load_model_and_tokenizer():
    print("Chargement du modèle et du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(DEVICE)
    return model, tokenizer


def reset_module_to_pretrained(model, layer_path, pretrained_model_name=PRETRAINED_MODEL_NAME):
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

    try:
        target_module = get_module(model, layer_path)
        source_module = get_module(base_model, layer_path)

        with torch.no_grad():
            for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):
                target_param.data.copy_(source_param.data)

        return True
    except Exception as e:
        return False


def reset_layers_to_pretrained(model, layer_indices):
    modified_model = copy.deepcopy(model)

    try:
        for layer_idx in layer_indices:
            layer_path = f"model.layers.{layer_idx}"
            success = reset_module_to_pretrained(modified_model, layer_path, PRETRAINED_MODEL_NAME)

        print(f"Nombre total de couches réinitialisées: {len(layer_indices)}")
        return modified_model
    except Exception as e:
        print(f"Erreur lors de la réinitialisation des couches: {e}")
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
    user_input = input("Entrez les indices des couches à réinitialiser: ")

    if user_input.strip():
        try:
            layer_indices = [int(idx.strip()) for idx in user_input.split(",")]
            print(f"Réinitialisation des couches aux indices: {layer_indices}")

            modified_model = reset_layers_to_pretrained(model, layer_indices)
            if modified_model is None:
                print("Échec de la réinitialisation des couches. Utilisation du modèle original pour le chat.")
                modified_model = model
        except ValueError as e:
            print(f"Erreur dans les indices fournis: {e}")
            print("Utilisation du modèle original pour le chat.")
            modified_model = model
    else:
        print("Aucune couche réinitialisée. Utilisation du modèle original pour le chat.")
        modified_model = model

    modified_model = modified_model.to(DEVICE)

    chat_with_model(modified_model, tokenizer)


if __name__ == "__main__":
    main()
