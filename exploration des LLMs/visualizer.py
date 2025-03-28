import json
import matplotlib.pyplot as plt


def plot_perplexity_ablation_results(file_path, log_scale=False):
    with open(file_path, 'r') as f:
        data = json.load(f)

    layers = []
    perplexities = []

    baseline = data["baseline"]["perplexity"]

    for key, value in data.items():
        if key.startswith("without_model.layers."):
            layer_num = int(key.split(".")[-1])
            layers.append(layer_num)
            perplexities.append(value["perplexity"])

    layers, perplexities = zip(*sorted(zip(layers, perplexities)))

    plt.figure(figsize=(10, 5))
    plt.plot(layers, perplexities, marker='o', linestyle='-', label='Perplexity per layer ablation')
    plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline perplexity')
    plt.xlabel("Layer removed")
    plt.ylabel("Perplexity")
    plt.title("Effect of Layer Ablation on Perplexity")
    plt.legend()
    plt.grid()

    if log_scale:
        plt.yscale('log')

    output_path = file_path.replace(".json", ".png")
    plt.savefig(output_path)


if __name__ == "__main__":
    file_path = "gemmametrics_reset/all_reset_results.json"
    plot_perplexity_ablation_results(file_path, log_scale=True)
