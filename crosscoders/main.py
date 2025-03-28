from nnsight import LanguageModel
from nnsight.intervention import base
import torch as th
from crosscoder import CrossCoder
from experiment import Experiment
import os
from args import get_config

dtype = th.float16

config = get_config()

print(config)

base_model = LanguageModel(config["base_model_name"], dispatch=True, torch_dtype=dtype)
chat_model = LanguageModel(config["chat_model_name"], dispatch=True, torch_dtype=dtype)

hsize = base_model.model.config.hidden_size
num_layers = 28

force_cpu = True
device = "cuda" if th.cuda.is_available() and not force_cpu else "cpu"


exp = Experiment(
    base_model,
    chat_model,
    CrossCoder.from_pretrained(
        "Butanium/crosscoder-Qwen2.5-0.5B-Instruct-and-Base-32k5e-2-50M-toks-73L0-0.84FVE",
        from_hub=True,
    ),
    14,
    crosscoder_device=device,
    max_acts=config["max_activations"],
)

html = exp.run(
    config["prompt"],
    config["features_compute"],
    config["highlight_features"],
    config["tooltip_features"],
)

fig, _ = exp.plot_norm_hist()

if os.path.exists(config["output_file"]):
    os.remove(config["output_file"])
if os.path.exists(config["output_plot"]):
    os.remove(config["output_plot"])

with open(config["output_file"], "w") as f:
    f.write(html)

fig.savefig(config["output_plot"])
