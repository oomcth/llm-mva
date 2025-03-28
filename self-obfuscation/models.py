"""Model loading and patching utilities for self-obfuscation experiments."""

import torch as th
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from obf_reps.models.hf import HFHardPromptedWithSelectableLayers
from obf_reps.models import ModelConfig

def load_local_model(
    checkpoint_path: Optional[str] = None,
    model_name: str = "google/gemma-2-9b-it",
    device=None,
):
    """Load a HF model, optionally from a checkpoint.

    Args:
        checkpoint_path: Optional path to a model checkpoint. If None, loads base model.
        model_name: Name of the base model.
        device: Device to load the model on. If None, uses default from ModelConfig.

    Returns:
        The loaded model.
    """
    print("Loading model...")
    if device is None:
        model_config = ModelConfig()
    else:
        model_config = ModelConfig(device=device)

    if checkpoint_path is not None:
        # Load model weights from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=model_config.device,
            torch_dtype=model_config.model_dtype,
        )
        model.eval()
        if not model_config.requires_grad:
            for p in model.parameters():
                p.requires_grad = False

        # Load and configure tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token:
            pass
        elif tokenizer.unk_token:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Create the wrapped model
        model = HFHardPromptedWithSelectableLayers(
            model=model, tokenizer=tokenizer, config=model_config
        )
    else:
        # Load everything from base model
        model = HFHardPromptedWithSelectableLayers.load_model(model_name, model_config)

    return model


def stitch_model(target_model, source_model, layers):
    """Replace specific layers in the target model with layers from the source model.
    
    Args:
        target_model: The model to modify.
        source_model: The model to take layers from.
        layers: List of layer indices to replace.
        
    Returns:
        The modified target model.
    """
    if not hasattr(target_model, "original_layers"):
        target_model.original_layers = [
            layer for layer in target_model.model.model.layers
        ]
    for layer in layers:
        target_model.model.model.layers[layer] = source_model.model.model.layers[layer]
    return target_model


def unstitch_model(target_model, layers=None):
    """Restore original layers in the target model.
    
    Args:
        target_model: The model to restore.
        layers: List of layer indices to restore. If None, restores all layers.
        
    Returns:
        The restored target model.
    """
    if not hasattr(target_model, "original_layers"):
        return target_model
        
    if layers is None:
        layers = list(range(len(target_model.original_layers)))
    for layer in layers:
        target_model.model.model.layers[layer] = target_model.original_layers[layer]
    return target_model


@th.no_grad()
def get_model_reps(model, entry, layers=None):
    """Get model representations for a given example.
    
    Args:
        model: The model to get representations from.
        entry: Dict-like object with 'prompt' and 'response' fields.
        layers: List of layer indices to get representations from.
        
    Returns:
        Model forward outputs containing representations.
    """
    if layers is None:
        layers = [12]  # Default to layer 12
        
    return model.forward_from_string(
        input_text=entry["prompt"],
        target_text=list(entry["response"]),
        add_chat_template=True,
        use_tunable_params=False,
        layers_to_probe=layers,
    )


def unembed_probs(reps, model):
    """Convert representations to token probabilities.
    
    Args:
        reps: Tensor of representations.
        model: The model to use for unembedding.
        
    Returns:
        Tensor of token probabilities.
    """
    assert reps.dim() <= 3
    unembed = model.model.lm_head(reps)
    return th.softmax(unembed, dim=-1)


def get_first_word_tokens_pos(tokenizer, prompt_or_tokens, word):
    """Find token positions of a specific word in tokenized text.
    
    Args:
        tokenizer: The tokenizer to use.
        prompt_or_tokens: The text or tokenized text to search in.
        word: The word to find.
        
    Returns:
        List of token positions corresponding to the word.
    """
    if isinstance(prompt_or_tokens, str):
        tokens = tokenizer.tokenize(prompt_or_tokens)
    elif len(prompt_or_tokens) == 0:
        return []
    elif not isinstance(prompt_or_tokens[0], str):
        tokens = tokenizer.convert_ids_to_tokens(prompt_or_tokens)
    else:
        tokens = prompt_or_tokens

    tokens_queue = tokens.copy()
    curr_prompt = ""
    last_idx = 0
    while word not in curr_prompt:
        if len(tokens_queue) == 0:
            raise ValueError(f"Word {word} not found in prompt {prompt_or_tokens}")
        curr_prompt += tokens_queue.pop(0)
        last_idx += 1
    curr_end_prompt = ""
    word_token_pos = []
    for i in range(last_idx - 1, -1, -1):
        curr_end_prompt = tokens[i] + curr_end_prompt
        word_token_pos.append(i)
        if word in curr_end_prompt:
            break
    else:
        raise ValueError(
            f"Error while finding word {word} in prompt {prompt_or_tokens}"
        )
    return word_token_pos 