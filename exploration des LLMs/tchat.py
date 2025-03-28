from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys


def chat_with_model(model_name):
    try:
        print(f"Chargement du modèle : {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        device = "mps"
        model.to(device)

        conversation_history = []

        while True:
            user_input = input("\nVous : ")

            if user_input.lower() in ['exit', 'quitter']:
                print("Fin de la conversation.")
                break

            conversation_history.append(f"Utilisateur : {user_input}")

            context = "\n".join(conversation_history) + "\nAssistant :"
            inputs = tokenizer(context, return_tensors="pt").to(device)

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if "Assistant :" in response:
                    assistant_response = response.split("Assistant :")[-1].strip()
                else:
                    assistant_response = response.strip()

                print(f"\nAssistant : {assistant_response}")

                conversation_history.append(f"Assistant : {assistant_response}")

            except Exception as e:
                print(f"Erreur lors de la génération de la réponse : {str(e)}")
                break

    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {str(e)}")
        print("Assurez-vous que le modèle est disponible sur Hugging Face et que vous avez les autorisations nécessaires.")


def main():
    default_model = "google/gemma-3-1b-it"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = default_model

    print("Démarrage de la conversation")
    chat_with_model(model_name)


if __name__ == "__main__":
    main()
