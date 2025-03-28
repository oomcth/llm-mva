import torch as th
from nnterp.nnsight_utils import get_layer_output, get_layer
from tiny_dashboard.html_utils import (
    create_highlighted_tokens_html,
    create_example_html,
    create_base_html,
)
from plots import plot_norm_hist


class Experiment:

    def __init__(
        self,
        base_model,
        instruct_model,
        crosscoder,
        collect_layer: int,
        window_size: int = 50,
        crosscoder_device: str | None = None,
        max_acts: dict[int, float] | None = None,
    ):
        """
        Args:
            base_model: Base language model
            instruct_model: Instruction-tuned model
            crosscoder: Model that combines base and instruct activations
            collect_layer: Layer to collect activations from
            window_size: Number of tokens to show before/after max activation
            crosscoder_device: Optional device to move crosscoder inputs to
        """
        self.base_model = base_model
        self.instruct_model = instruct_model
        self.tokenizer = instruct_model.tokenizer
        self.crosscoder = crosscoder
        self.window_size = window_size
        self.crosscoder_device = crosscoder_device
        self.layer = collect_layer
        self.max_acts = max_acts

    def plot_norm_hist(self):
        fig, ax = plot_norm_hist(
            self.crosscoder, "32k latents, 5e-2 L1 penalty, 50M tokens"
        )
        return fig, ax

    @th.no_grad()
    def get_feature_activation(
        self, text: str, feature_indices: tuple[int, ...]
    ) -> th.Tensor:
        """Get the activation values for given features by combining base and instruct model activations"""
        with self.instruct_model.trace(
            text
        ):  # self.model is the instruct_model from parent
            instruct_activations = get_layer_output(self.instruct_model, self.layer)[
                0
            ].save()
            get_layer(self.instruct_model, self.layer).output.stop()

        with self.base_model.trace(text):
            base_activations = get_layer_output(self.base_model, self.layer)[0].save()
            get_layer(self.base_model, self.layer).output.stop()

        if self.crosscoder_device is not None:
            base_activations = base_activations.to(self.crosscoder_device)
            instruct_activations = instruct_activations.to(self.crosscoder_device)

        cc_input = th.stack([base_activations, instruct_activations], dim=1).float()
        features_acts = self.crosscoder.get_activations(
            cc_input, select_features=list(feature_indices)
        )
        return features_acts

    def _create_html_highlight(
        self,
        tokens: list[str],
        activations: th.Tensor,
        all_feature_indices: list[int],
        highlight_features: list[int],
        tooltip_features: list[int],
        return_max_acts: bool = False,
    ) -> str | tuple[str, str]:
        """Create HTML with highlighted tokens based on activation values"""
        # Map feature indices to their positions in the activations tensor
        highlight_positions = [all_feature_indices.index(f) for f in highlight_features]
        tooltip_positions = [all_feature_indices.index(f) for f in tooltip_features]

        # Create feature names mapping indices to their original feature numbers
        activation_names = [
            f"Feature {all_feature_indices[i]}" for i in range(len(all_feature_indices))
        ]

        # Handle min_max_act value
        min_max_act = None
        # min_max_act_value = self.min_max_act_input.value.strip().lower()
        min_max_act_value = ""

        if min_max_act_value == "":
            min_max_act = None
        elif min_max_act_value != "auto":
            try:
                min_max_act = float(min_max_act_value)
            except ValueError:
                raise ValueError("Min-max act must be empty, 'auto' or a float value")
        elif min_max_act_value == "auto" and self.max_acts is not None:
            # Use the first highlight feature's max_act value
            feature = highlight_features[0]
            if feature in self.max_acts:
                min_max_act = self.max_acts[feature]
            else:
                raise ValueError(f"No max activation value found for feature {feature}")
        elif min_max_act_value == "auto":
            raise ValueError(
                "Cannot use 'auto' without max_acts dictionary provided during initialization"
            )

        return create_highlighted_tokens_html(
            tokens=tokens,
            activations=activations,
            tokenizer=self.tokenizer,
            highlight_features=highlight_positions,
            tooltip_features=tooltip_positions,
            color1=(255, 0, 0),
            color2=(0, 255, 0),
            activation_names=activation_names,
            return_max_acts_str=return_max_acts,
            min_max_act=min_max_act,
        )

    def run(self, text, feature_indices, highlight_features, tooltip_features):
        features = list(
            dict.fromkeys(highlight_features + tooltip_features + feature_indices)
        )

        tokens = self.tokenizer.tokenize(text, add_special_tokens=True)

        act = self.get_feature_activation(text, tuple(features))

        assert len(tokens) == act.shape[0]

        html_content, max_acts_str = self._create_html_highlight(
            tokens,
            act,
            features,
            highlight_features,
            tooltip_features,
            return_max_acts=True,
        )

        example_html = create_example_html(max_acts_str, html_content, static=True)

        html = create_base_html(title="hello", content=example_html)

        # print(example_html)
        return html
