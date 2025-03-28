import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_dataset
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_activations(model, tokenizer, dataset, layer_idx):

    activations = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    for example in tqdm(dataset):
        #text = example['messages'][0]['content']
        text = example['ctx']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        layer_activations = torch.stack(outputs.hidden_states)
        #print(layer_activations.shape)
        #mean_activations = layer_activations.mean(dim=2).cpu().numpy()
        mean_activations = layer_activations[:,:,-1].cpu().numpy()
        activations.append(mean_activations)
    
    return np.concatenate(activations, axis=1)

# def get_activations(model, tokenizer, dataset, layer_idx=None, max_length=512):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     #inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)
#     model.to(device)
#     model.eval()
#     layers_activations = [[] for _ in range(len(model.model.layers))]

#     def hook_f(i):
#         def hook_fn(module, input, output):
#             layers_activations[i].append(output[0][0,-1].detach().cpu().numpy())
#         return hook_fn

#     hooks = []
#     for i, layer in enumerate(model.model.layers):
#         hook = layer.register_forward_hook(hook_f(i))
#         hooks.append(hook)

#     for example in tqdm(dataset):
#         #text = example['messages'][0]['content']
#         text = example['ctx']
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

#         with torch.no_grad():
#             model(**inputs)

#     for hook in hooks:
#         hook.remove()

#     #print(layers_activations[0][0].shape)

#     layers_activations = [np.stack(act) for act in layers_activations]
#     layers_activations = np.stack(layers_activations)
#     #print(layers_activations.shape)

#     return layers_activations


def plot_pca_difference(activation_differences):
    n_layers = activation_differences.shape[0]
    print(activation_differences.shape)
    for i in range(n_layers):
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(activation_differences[i])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.5)
        plt.title(f'PCA of Activations Differences layer {i}  (Finetuned - Pretrained)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.scatter(transformed[:, 0], transformed[:, 1])
        plt.savefig(f'pca_layer_{i}.png')


def main():
    num_samples = 100
    dataset_name = 'lmsys/lmsys-chat-1m'
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]", token=hf_token)

    pretrained_name = "google/gemma-3-1b-pt"
    finetuned_name = "google/gemma-3-1b-it"
    
    pretrained_model = Gemma3ForCausalLM.from_pretrained(
        pretrained_name,
        output_hidden_states=True,
        token=hf_token,
        device_map="auto"
    )
    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_name, token=hf_token)

    finetuned_model = Gemma3ForCausalLM.from_pretrained(
        finetuned_name,
        output_hidden_states=True,
        token=hf_token,
        device_map="auto"
    )
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_name, token=hf_token)

    layer_idx = 16

    pretrained_acts = get_activations(pretrained_model, pretrained_tokenizer, dataset, layer_idx)
    finetuned_acts = get_activations(finetuned_model, finetuned_tokenizer, dataset, layer_idx)
    activation_differences = finetuned_acts - pretrained_acts

    plot_pca_difference(activation_differences)


if __name__ == "__main__":
    main()
