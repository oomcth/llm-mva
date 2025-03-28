"""Probe loading and evaluation functions for self-obfuscation experiments."""

import os
import pickle
import torch as th
from typing import Dict, List, Optional
from pathlib import Path

from obf_reps.metrics import TrainableMetric

from models import get_model_reps


def load_word_to_probe_dict(probe_checkpoint_dir: str) -> Dict[str, TrainableMetric]:
    """
    Load a dictionary mapping words to their corresponding probe objects from a given directory.

    The function assumes that each probe is saved as a .pkl file in the specified directory,
    where the filename (without the .pkl extension) is the word. For example, a file named
    "bomb.pkl" will be loaded as the probe for the word "bomb".

    Args:
        probe_checkpoint_dir (str): Path to the directory containing the probe checkpoint files.

    Returns:
        dict[str, TrainableMetric]: A dictionary mapping word (str) to the loaded probe object.
    """
    if not os.path.isdir(probe_checkpoint_dir):
        raise ValueError(f"Provided path is not a directory: {probe_checkpoint_dir}")

    word_to_probe: Dict[str, TrainableMetric] = {}
    # Iterate over all files in the directory
    for filename in os.listdir(probe_checkpoint_dir):
        if filename.endswith(".pkl"):
            word = filename[:-4]  # Remove the ".pkl" extension to get the word
            word = word.replace("train_", "").replace("test_", "")
            file_path = os.path.join(probe_checkpoint_dir, filename)
            with open(file_path, "rb") as f:
                probe = pickle.load(f)
            if not isinstance(probe, TrainableMetric):
                raise ValueError(f"Loaded object is not a TrainableMetric: {file_path}")
            word_to_probe[word] = probe

    return word_to_probe


@th.no_grad()
def get_probe_score(probe, entry, model):
    """Calculate probe score for a given example.
    
    Args:
        probe: The probe to evaluate with.
        entry: Dict-like object with prompt and response.
        model: The model to get representations from.
        
    Returns:
        Tensor containing probe scores.
    """
    forward_return = get_model_reps(model, entry)
    return probe.predict_example(
        input_reps=forward_return.input_reps,
        target_reps=forward_return.target_reps,
        target_mask=forward_return.loss_mask,
    )


@th.no_grad()
def get_dist_between_reps(entries, base_model, obf_model):
    """Calculate distance metrics between base and obfuscated model representations.
    
    Args:
        entries: Examples to evaluate on.
        base_model: The base model.
        obf_model: The obfuscated model.
        
    Returns:
        Tuple of various distance metrics.
    """
    layers = list(range(len(base_model.model.model.layers)))
    base_reps = get_model_reps(base_model, entries, layers=layers)
    obf_reps = get_model_reps(obf_model, entries, layers=layers)

    # Input representation norms
    input_base_reps_norms = (
        th.norm(base_reps.input_reps, dim=-1)
        .transpose(0, 1)
        .reshape(len(layers), -1)
        .mean(dim=-1)
    )
    input_obf_reps_norms = (
        th.norm(obf_reps.input_reps, dim=-1)
        .transpose(0, 1)
        .reshape(len(layers), -1)
        .mean(dim=-1)
    )
    input_reps_norms = (
        th.norm(base_reps.input_reps - obf_reps.input_reps, dim=-1)
        .transpose(0, 1)
        .reshape(len(layers), -1)
        .mean(dim=-1)
    )
    input_reps_norms_normalized = 2 * (
        th.norm(base_reps.input_reps - obf_reps.input_reps, dim=-1)
        / (th.norm(base_reps.input_reps, dim=-1) + th.norm(obf_reps.input_reps, dim=-1))
    ).transpose(0, 1).reshape(len(layers), -1).mean(dim=-1)

    # Target representation norms
    target_base_reps_norms = (
        th.norm(base_reps.target_reps, dim=-1)
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )
    target_obf_reps_norms = (
        th.norm(obf_reps.target_reps, dim=-1)
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )
    target_reps_norms = (
        th.norm(base_reps.target_reps - obf_reps.target_reps, dim=-1)
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )
    target_reps_norms_normalized = (
        (
            2
            * th.norm(base_reps.target_reps - obf_reps.target_reps, dim=-1)
            / (
                th.norm(base_reps.target_reps, dim=-1)
                + th.norm(obf_reps.target_reps, dim=-1)
            )
        )
        .transpose(1, 2)[base_reps.loss_mask]
        .mean(dim=0)
    )
    
    # KL divergence metrics
    kl_div_input = th.nn.functional.kl_div(
        th.log_softmax(obf_reps.input_logits, dim=-1),
        th.log_softmax(base_reps.input_logits, dim=-1),
        log_target=True,
        reduction="batchmean",
    )
    kl_div_target = (
        th.nn.functional.kl_div(
            th.log_softmax(obf_reps.target_logits[obf_reps.loss_mask], dim=-1),
            th.log_softmax(base_reps.target_logits[base_reps.loss_mask], dim=-1),
            log_target=True,
            reduction="sum",
        )
        / base_reps.loss_mask.sum()
    )
    
    # Statistical distance metrics
    base_probs_input = th.softmax(base_reps.input_logits, dim=-1)
    obf_probs_input = th.softmax(obf_reps.input_logits, dim=-1)
    base_probs_target = th.softmax(base_reps.target_logits, dim=-1)
    obf_probs_target = th.softmax(obf_reps.target_logits, dim=-1)
    stat_dist_input = (base_probs_input - obf_probs_input).abs().sum(dim=-1).mean()
    stat_dist_target = (base_probs_target - obf_probs_target).abs().sum(dim=-1).mean()
    
    return (
        input_reps_norms.cpu().numpy(),
        target_reps_norms.cpu().numpy(),
        input_reps_norms_normalized.cpu().numpy(),
        target_reps_norms_normalized.cpu().numpy(),
        input_base_reps_norms.cpu().numpy(),
        input_obf_reps_norms.cpu().numpy(),
        target_base_reps_norms.cpu().numpy(),
        target_obf_reps_norms.cpu().numpy(),
        kl_div_input.item(),
        kl_div_target.item(),
        stat_dist_input.item(),
        stat_dist_target.item(),
    ) 