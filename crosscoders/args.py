import argparse
from tiny_dashboard.utils import parse_list_str
import os
import sys

def get_config():
    parser = argparse.ArgumentParser(description='Run a crosscoder experiment')

    parser.add_argument('--base_model_name', type=str, default="Qwen/Qwen2.5-0.5B", help='The base model name')
    parser.add_argument('--chat_model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help='The chat model name')
    parser.add_argument('--output_file', type=str, default='output.html', help='The output file')
    parser.add_argument('--output_plot', type=str, default='output_plot.png', help='The output plot')
    parser.add_argument('--prompt', type=str, default='What is the capital of France?', help='The prompt')
    parser.add_argument('--features_compute', type=str, help='The features to compute')
    parser.add_argument('--highlight_features', type=str, help='The features to highlight')
    parser.add_argument('--tooltip_features', type=str, help='The features to tooltip')
    parser.add_argument('--max_activations', type=float, default=None, help='The max activations')

    args = parser.parse_args()

    config = {
        'base_model_name': args.base_model_name,
        'chat_model_name': args.chat_model_name,
        # 'output_file': os.path.join(ROOT, args.output_file),
        'output_file': args.output_file,
        'output_plot': args.output_plot,
        'prompt': args.prompt,
        'features_compute': features_compute_list,
        'highlight_features': highlight_features_list,
        'tooltip_features': tooltip_features_list,
        'max_activations': args.max_activations,
    }

    return config
