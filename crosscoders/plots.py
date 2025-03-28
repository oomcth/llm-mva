# %%
import torch as th
from crosscoder import CrossCoder

# %%
from crosscoder import CrossCoder
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch as th
from tqdm.auto import tqdm

plt.rcParams["text.usetex"] = False
plt.rcParams.update({"font.size": 20})
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


@th.no_grad()
def plot_norm_hist(crosscoder, name, fig=None, ax=None):
    norms = crosscoder.decoder.weight.norm(dim=2)
    rel_norms = 0.5 * ((norms[1] - norms[0]) / th.maximum(norms[0], norms[1]) + 1)
    values = rel_norms.detach().cpu().numpy()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.0))

    hist, bins, _ = ax.hist(
        values, bins=100, color="lightgray", label="Other", log=True
    )
    ax.hist(values, bins=bins, color="lightgray", log=True)  # Base gray histogram
    ax.hist(
        values[((values >= 0.4) & (values < 0.6))],
        bins=bins,
        color="C1",
        label="Shared",
        log=True,
    )
    ax.hist(
        values[((values >= 0.9))], bins=bins, color="C0", label="Chat-only", log=True
    )
    ax.hist(
        values[(values <= 0.1)],
        bins=bins,
        color="limegreen",
        label="Base-only",
        log=True,
    )
    ax.set_xticks([0, 0.1, 0.4, 0.5, 0.6, 0.9, 1])
    ax.axvline(x=0.1, color="green", linestyle="--", alpha=0.5)
    ax.axvline(x=0.4, color="C1", linestyle="--", alpha=0.5)
    ax.axvline(x=0.6, color="C1", linestyle="--", alpha=0.5)
    ax.axvline(x=0.9, color="C0", linestyle="--", alpha=0.5)
    ax.set_xlabel("Relative Norm Difference")
    ax.set_ylabel("Latents")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title(name)
    fig.tight_layout()
    return fig, ax


def path_from_params(
    save_path="/scratch/cdumas/mva/models",
    width=16,
    tokens="latest",
    L1_penalty="5e-2",
    run_name="1741001799_ultramarine-camel",
):
    if tokens != "latest":
        file = str(tokens) + "_toks.pt"
    else:
        file = "latest.pt"
    path = f"{save_path}/{run_name}/{run_name}_{width}k{L1_penalty}/{file}"
    return path


def crosscoder_from_path(
    save_path="/scratch/cdumas/mva/models",
    width=16,
    tokens="latest",
    L1_penalty="5e-2",
    run_name="1741001799_ultramarine-camel",
):
    path = path_from_params(save_path, width, tokens, L1_penalty, run_name)
    return CrossCoder.from_pretrained(path)


# %%
def plot_norm_hist_grid(
    model_dir, run_name, tokens=None, widths=None, L1_penalties=None
):
    if tokens is None:
        tokens = [1000448, 2000896, 3001344]
    if widths is None:
        widths = ["16k", "32k"]
    if L1_penalties is None:
        L1_penalties = ["5e-2", "3e-2"]

    from itertools import product

    all_settings = list(product(tokens, widths, L1_penalties))

    num_rows = len(tokens)
    num_cols = len(widths) * len(L1_penalties)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(34, 16))
    from tqdm import tqdm

    for (tokens, width, L1_penalty), ax in tqdm(
        zip(all_settings, axes.flatten()), total=len(all_settings)
    ):
        path = f"{model_dir}/{run_name}/{run_name}_{width}{L1_penalty}/{tokens}_toks.pt"
        model = CrossCoder.from_pretrained(path)
        tok_string = f"{tokens // 1e6}M tokens"
        plot_norm_hist(
            model,
            f"{width} latents, {L1_penalty} sparsity penalty, {tok_string}",
            ax=ax,
            fig=fig,
        )

    plt.savefig("norm_hist.png", dpi=300)
    return fig, axes

if __name__ == "__main__":
    _ = plot_norm_hist_grid(
        "/scratch/cdumas/mva/tests/models",
        "1741043371_vegan-ladybug",
        tokens=[3_001_344, 10_004_480, 34_015_232, 50_000_896],
    )

# %%

    path = path_from_params(
        save_path="/scratch/cdumas/mva/tests/models",
        run_name="1741043371_vegan-ladybug",
        tokens = 50000896,
        width=32,
        L1_penalty="5e-2",
    )
    model = CrossCoder.from_pretrained(path)
    # %%
    from huggingface_hub import HfApi

    repo_id = "Butanium/crosscoder-Qwen2.5-0.5B-Instruct-and-Base-32k5e-2-50M-toks-73L0-0.84FVE"
    api = HfApi()

    api.create_repo(repo_id=repo_id, repo_type="model")

    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="pytorch_model.bin",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Added crosscoder weights",
    )
    # %%
    import tempfile
    import json
    import os

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "config.json")
        json_dict = {
            "model_type": "crosscoder",
            "model_0": "Qwen2.5-0.5B",
            "model_1": "Qwen2.5-0.5B-Instruct",
            "activation_dim": model.decoder.weight.shape[2],
            "dict_size": model.decoder.weight.shape[1],
            "num_layers": 2,
            "mu": 5e-2,
            "learning_rate": 1e-4,
            "batch_size": 1024,
            "dataset": "lmsys/lmsys-chat-1m",
            "num_tokens": 50_000_896,
            "width": 32000,
            "L1_penalty": 5e-2,
            "l0_validation": 73,
            "frac_var_explained_validation": 0.84,
            "dead_latents_validation": 1050,
        }
        with open(path, "w") as f:
            json.dump(json_dict, f)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Added crosscoder config",
        )
    # %%
    # %%
    from crosscoder import CrossCoder

    hf_model = CrossCoder.from_pretrained(
        "Butanium/crosscoder-Qwen2.5-0.5B-Instruct-and-Base-32k5e-2-50M-toks-73L0-0.84FVE", from_hub=True
    )
    # %%
