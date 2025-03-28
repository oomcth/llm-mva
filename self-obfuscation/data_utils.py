"""Data loading and processing functions for self-obfuscation experiments."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_test_data(data_path: str) -> pd.DataFrame:
    """
    Load test data from a JSON file.
    
    Args:
        data_path: Path to JSON file with test data.
        
    Returns:
        DataFrame containing the test data.
    """
    with open(data_path, "r") as f:
        test_data = json.load(f)["data"]
    
    return pd.DataFrame(test_data)


def create_example_variations(test_data_df: pd.DataFrame, words: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Create variations of examples for different words and adversarial settings.
    
    Args:
        test_data_df: DataFrame with test data.
        words: List of words to create variations for.
        
    Returns:
        Dictionary mapping example variation names to corresponding DataFrames.
    """
    entries = {}

    # Generate all combinations of word and adversarial word
    for word in words:
        # Get base examples for the word
        base_examples = test_data_df.query(f"word == '{word}'")
        
        # Add vanilla examples (non-adversarial)
        vanilla_examples = base_examples.query("adversarial_word.isna()")
        entries[f"{word.capitalize()} vanilla"] = vanilla_examples
        
        # Add adversarial examples for each target word
        for adv_word in words:
            if word == adv_word:
                # For cases where the word is the same as the adversarial word
                entries[f"{word.capitalize()} adv {adv_word}"] = base_examples.query(
                    f"adversarial_word == '{adv_word}'"
                )
            else:
                # For cases where we need to create cross-adversarial examples
                source_examples = base_examples.query(f"adversarial_word == '{word}'")
                # Copy the examples and replace the adversarial word in the prompt
                modified_examples = source_examples.copy()
                modified_examples["prompt"] = modified_examples["prompt"].apply(
                    lambda x: x.replace(
                        f"You are being probed for '{word}'",
                        f"You are being probed for '{adv_word}'",
                    )
                )
                entries[f"{word.capitalize()} adv {adv_word}"] = modified_examples

    return entries


def get_test_words():
    """
    Get list of test words used in experiments.
    
    Returns:
        List of test words.
    """
    return [
        "gun",
        "apple",
        "elephant",
        "SQL",
        "cloud",
        "jester",
        "mathematical",
        "samantha",
    ]