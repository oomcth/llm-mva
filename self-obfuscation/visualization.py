"""Visualization utilities for self-obfuscation experiments."""

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_layer_sweep_results(layer_sweep_results, max_layer, save_dir="results"):
    """Plot layer sweep results for different interventions.
    
    Args:
        layer_sweep_results: Dictionary of layer sweep results.
        max_layer: Maximum layer index.
        save_dir: Directory to save plots to.
        
    Returns:
        Plotly figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    fig = make_subplots(
        rows=1,
        cols=2,
        x_title="Layer",
        subplot_titles=["Patch Obfuscated → Base", "Patch Base → Obfuscated"],
    )

    # Define styling parameters
    colors = {"adv_gun": "green", "vanilla_gun": "purple"}
    dash_styles = {1: None, 3: "dash", 5: "longdashdot"}
    marker_styles = {1: "circle", 3: "square", 5: "diamond"}
    
    # Plot layer sweep results
    for col, from_model in enumerate(["obf", "base"]):
        for num_layers in [1, 3, 5]:
            for example_type in ["adv_gun", "vanilla_gun"]:
                data_key = f"{num_layers}_layer_from_{from_model}_{example_type}"
                if data_key in layer_sweep_results:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(max_layer + 1)),
                            y=layer_sweep_results[data_key],
                            mode="lines+markers",
                            name=f"{num_layers} layers - {'Adversarial' if example_type == 'adv_gun' else 'Vanilla'} Gun Score",
                            line=dict(color=colors[example_type], dash=dash_styles[num_layers]),
                            showlegend=col == 0,
                            marker=dict(symbol=marker_styles[num_layers]),
                        ),
                        row=1,
                        col=col + 1,
                    )

    # Update layout
    fig.update_layout(
        title="Layer Sweep Results",
        xaxis_title="Layer",
        yaxis_title="Probe Score",
        legend_title="Metrics",
        width=1200,
        height=800,
    )
    
    # Save figures
    fig.write_html(f"{save_dir}/layer_sweep_results.html")
    fig.write_image(f"{save_dir}/layer_sweep_results.png", width=1200, height=800, scale=2)
    
    return fig


def plot_stitching_results(
    res_adv_gun, 
    res_vanilla_gun, 
    res_adv_gun_rev, 
    res_vanilla_gun_rev,
    layers_5_6_adv_gun,
    layers_5_6_vanilla_gun,
    max_layer,
    save_dir="results"
):
    """Plot results of forward and reverse layer stitching experiments.
    
    Args:
        res_adv_gun: List of forward stitching results for adversarial gun examples.
        res_vanilla_gun: List of forward stitching results for vanilla gun examples.
        res_adv_gun_rev: List of reverse stitching results for adversarial gun examples.
        res_vanilla_gun_rev: List of reverse stitching results for vanilla gun examples.
        layers_5_6_adv_gun: Result for layers 5-6 for adversarial gun examples.
        layers_5_6_vanilla_gun: Result for layers 5-6 for vanilla gun examples.
        max_layer: Maximum layer index.
        save_dir: Directory to save plots to.
        
    Returns:
        Plotly figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    fig = go.Figure()

    # Forward stitching traces
    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer + 1)),
            y=res_adv_gun,
            mode="lines+markers",
            name="Forward - Adversarial Gun Score",
            line=dict(color="red"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer + 1)),
            y=res_vanilla_gun,
            mode="lines+markers",
            name="Forward - Vanilla Gun Score",
            line=dict(color="blue"),
        )
    )

    # Reverse stitching traces
    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer, -1, -1)),
            y=res_adv_gun_rev,
            mode="lines+markers",
            name="Reverse - Adversarial Gun Score",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(max_layer, -1, -1)),
            y=res_vanilla_gun_rev,
            mode="lines+markers",
            name="Reverse - Vanilla Gun Score",
            line=dict(color="blue", dash="dash"),
        )
    )

    # Add horizontal lines for layer 5-6 intervention
    fig.add_shape(
        type="line",
        x0=0,
        y0=layers_5_6_adv_gun,
        x1=max_layer,
        y1=layers_5_6_adv_gun,
        line=dict(
            color="red",
            width=2,
            dash="dot",
        ),
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=layers_5_6_vanilla_gun,
        x1=max_layer,
        y1=layers_5_6_vanilla_gun,
        line=dict(
            color="blue",
            width=2,
            dash="dot",
        ),
    )

    # Add annotations for the intervention lines
    fig.add_annotation(
        x=max_layer,
        y=layers_5_6_adv_gun,
        text="Layer 5-6 Adversarial",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=0,
        font=dict(color="red", size=12),
    )

    fig.add_annotation(
        x=max_layer,
        y=layers_5_6_vanilla_gun,
        text="Layer 5-6 Vanilla",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=0,
        font=dict(color="blue", size=12),
    )

    # Update layout
    fig.update_layout(
        title="Gun Probe Scores with Layer Stitching",
        xaxis_title="Layer",
        yaxis_title="Probe Score",
        height=800,
        width=1200,
    )
    
    # Save figures
    fig.write_html(f"{save_dir}/gun_probe_scores_with_layer_stitching.html")
    fig.write_image(f"{save_dir}/gun_probe_scores_with_layer_stitching.png", width=1200, height=800, scale=2)
    
    return fig


def plot_representation_norms(
    input_reps_norms_adv_gun,
    target_reps_norms_adv_gun,
    input_reps_norms_normalized_adv_gun,
    target_reps_norms_normalized_adv_gun,
    input_base_reps_norms_adv_gun,
    input_obf_reps_norms_adv_gun,
    input_reps_norms_vanilla_gun,
    target_reps_norms_vanilla_gun,
    input_reps_norms_normalized_vanilla_gun, 
    target_reps_norms_normalized_vanilla_gun,
    input_base_reps_norms_vanilla_gun,
    input_obf_reps_norms_vanilla_gun,
    save_dir="results"
):
    """Plot representation norm analysis.
    
    Args:
        Various representation norm metrics for adversarial and vanilla gun examples.
        save_dir: Directory to save plots to.
        
    Returns:
        Matplotlib figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # Left subplot - Raw values
    ax1.plot(
        input_reps_norms_adv_gun,
        label="Input reps (adversarial gun)",
        marker="o",
        color="red",
        linewidth=2,
        linestyle="-",
    )
    ax1.plot(
        input_reps_norms_vanilla_gun,
        label="Input reps (vanilla gun)",
        marker="x",
        color="blue",
        linewidth=2,
        linestyle="-",
    )
    ax1.plot(
        target_reps_norms_adv_gun,
        label="Target reps (adversarial gun)",
        marker="s",
        color="darkred",
        linewidth=2,
        linestyle="--",
    )
    ax1.plot(
        target_reps_norms_vanilla_gun,
        label="Target reps (vanilla gun)",
        marker="+",
        color="darkblue",
        linewidth=2,
        linestyle="--",
    )
    ax1.set_title("Raw Representation Difference Norms", fontsize=14)
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Norm of the Difference", fontsize=12)

    # Middle subplot - Normalized values
    ax2.plot(
        input_reps_norms_normalized_adv_gun,
        label="Input reps (adversarial gun)",
        marker="o",
        color="red",
        linewidth=2,
        linestyle="-",
    )
    ax2.plot(
        input_reps_norms_normalized_vanilla_gun,
        label="Input reps (vanilla gun)",
        marker="x",
        color="blue",
        linewidth=2,
        linestyle="-",
    )
    ax2.plot(
        target_reps_norms_normalized_adv_gun,
        label="Target reps (adversarial gun)",
        marker="s",
        color="darkred",
        linewidth=2,
        linestyle="--",
    )
    ax2.plot(
        target_reps_norms_normalized_vanilla_gun,
        label="Target reps (vanilla gun)",
        marker="+",
        color="darkblue",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_title("Normalized Representation Difference Norms", fontsize=14)
    ax2.set_xlabel("Layer Index", fontsize=12)
    ax2.set_ylabel("Normalized Norm of the Difference", fontsize=12)

    # Right subplot - Raw norms from both models
    ax3.plot(
        input_base_reps_norms_adv_gun,
        label="Base model input (adversarial gun)",
        marker="o",
        color="red",
        linewidth=2,
        linestyle="-",
    )
    ax3.plot(
        input_base_reps_norms_vanilla_gun,
        label="Base model input (vanilla gun)",
        marker="x",
        color="blue",
        linewidth=2,
        linestyle="-",
    )
    ax3.plot(
        input_obf_reps_norms_adv_gun,
        label="Obf model input (adversarial gun)",
        marker="s",
        color="darkred",
        linewidth=2,
        linestyle="--",
    )
    ax3.plot(
        input_obf_reps_norms_vanilla_gun,
        label="Obf model input (vanilla gun)",
        marker="+",
        color="darkblue",
        linewidth=2,
        linestyle="--",
    )
    ax3.set_title("Raw Representation Norms", fontsize=14)
    ax3.set_xlabel("Layer Index", fontsize=12)
    ax3.set_ylabel("Norm of Representations", fontsize=12)

    fig.suptitle("Representation Analysis: Base vs Self-Obfuscated Model", fontsize=16)

    # Add grid and legend to all subplots
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=9, loc="best")

    # Save figure
    plt.tight_layout()
    plt.savefig(f"{save_dir}/representation_norms_analysis.png", dpi=300)
    
    return fig


def plot_kl_divergence(
    kl_div_input_adv_gun,
    kl_div_target_adv_gun,
    stat_dist_input_adv_gun,
    stat_dist_target_adv_gun,
    kl_div_input_vanilla_gun,
    kl_div_target_vanilla_gun,
    stat_dist_input_vanilla_gun,
    stat_dist_target_vanilla_gun,
    save_dir="results"
):
    """Plot KL divergence and statistical distance metrics.
    
    Args:
        Various KL divergence and statistical distance metrics.
        save_dir: Directory to save plots to.
        
    Returns:
        Plotly figure object.
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    fig = make_subplots(rows=1, cols=2)

    # Add bars for KL divergence
    fig.add_trace(
        go.Bar(
            x=["Gun adv gun", "Gun vanilla"],
            y=[kl_div_input_adv_gun, kl_div_input_vanilla_gun],
            name="Input KL Divergence",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["Gun adv gun", "Gun vanilla"],
            y=[kl_div_target_adv_gun, kl_div_target_vanilla_gun],
            name="Target KL Divergence",
        ),
        row=1,
        col=1,
    )
    
    # Add bars for statistical distance
    fig.add_trace(
        go.Bar(
            x=["Gun adv gun", "Gun vanilla"],
            y=[stat_dist_input_adv_gun, stat_dist_input_vanilla_gun],
            name="Input Stat Dist",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=["Gun adv gun", "Gun vanilla"],
            y=[stat_dist_target_adv_gun, stat_dist_target_vanilla_gun],
            name="Target Stat Dist",
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title_text="KL Divergence between Base Model and Self-Obfuscated Model",
        xaxis_title="Example Type",
        yaxis_title="KL Divergence",
        height=800,
        width=1200,
        legend_title="Metric",
        barmode="group",  # Group bars side by side
    )
    
    # Save figures
    fig.write_html(f"{save_dir}/kl_divergence_between_base_and_self_obfuscated_model.html")
    fig.write_image(
        f"{save_dir}/kl_divergence_between_base_and_self_obfuscated_model.png",
        width=1200,
        height=800,
        scale=2,
    )
    
    return fig


def plot_top_tokens(probs, tokens, tokenizer, height=800, width=1250, title="Token Probabilities", annot_size=10):
    """Plot heatmap of top token probabilities.
    
    Args:
        probs: Token probabilities tensor.
        tokens: List of tokens.
        tokenizer: Tokenizer for converting token IDs to text.
        height, width: Plot dimensions.
        title: Plot title.
        annot_size: Size of annotation text.
        
    Returns:
        Plotly figure object.
    """
    # Get top-10 token indices and their probabilities for each position
    top_k = 10
    top_probs, top_indices = probs.topk(top_k, dim=-1)
    
    # Define the x-axis (token positions)
    x_pos = list(range(len(tokens)))
    
    # Initialize figure
    fig = go.Figure()
    
    # Add a heatmap for token probabilities
    for i, token_pos in enumerate(x_pos):
        # Convert token indices to strings
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices[i]]
        
        # Create a hovertext with token and probability
        hover_texts = [f"{token}: {prob:.4f}" for token, prob in zip(top_tokens, top_probs[i])]
        
        # Add probe scores as a heatmap row
        fig.add_trace(
            go.Heatmap(
                z=top_probs[i].unsqueeze(0),
                x=[f"{token_pos}_{j}" for j in range(top_k)],
                y=[tokens[i]],
                text=[[t] for t in top_tokens],
                hovertext=[[t] for t in hover_texts],
                colorscale='Viridis',
                showscale=(i == 0),  # Only show colorbar for the first heatmap
                colorbar=dict(title="Probability"),
            )
        )
    
    # Add annotations with token names
    for i, token_pos in enumerate(x_pos):
        for j in range(top_k):
            token = tokenizer.decode([top_indices[i][j]])
            prob = top_probs[i][j].item()
            
            # Add token annotation
            fig.add_annotation(
                x=f"{token_pos}_{j}",
                y=tokens[i],
                text=f"{token}<br>({prob:.4f})",
                showarrow=False,
                font=dict(size=annot_size, color="white" if prob > 0.5 else "black")
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Token Position & Rank",
        yaxis_title="Input Token",
        height=height,
        width=width,
    )
    
    # Disable hover on empty points
    fig.update_traces(hoverinfo="text")
    
    return fig 