import torch.nn as nn
import torch as th
from torch.nn.functional import relu
from warnings import warn
import einops
from huggingface_hub import PyTorchModelHubMixin

"""
Code inspired by https://github.com/saprmarks/dictionary_learning
"""


class Encoder(nn.Module):
    """
    A cross-coder encoder
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers=None,
        same_init_for_all_layers: bool = False,
        norm_init_scale: float | None = None,
        encoder_layers: list[int] | None = None,
    ):
        super().__init__()

        if encoder_layers is None:
            if num_layers is None:
                raise ValueError(
                    "Either encoder_layers or num_layers must be specified"
                )
            encoder_layers = list(range(num_layers))
        else:
            num_layers = len(encoder_layers)
        self.encoder_layers = encoder_layers
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        if same_init_for_all_layers:
            weight = nn.init.kaiming_uniform_(th.empty(activation_dim, dict_size))
            weight = weight.repeat(num_layers, 1, 1)
        else:
            weight = nn.init.kaiming_uniform_(
                th.empty(num_layers, activation_dim, dict_size)
            )
        if norm_init_scale is not None:
            weight = weight / weight.norm(dim=1, keepdim=True) * norm_init_scale
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(th.zeros(dict_size))

    def forward(
        self,
        x: th.Tensor,
        select_features: list[int] | None = None,
    ) -> th.Tensor:  # (batch_size, activation_dim)
        """
        Convert activations to features for each layer

        Args:
            x: (batch_size, n_layers, activation_dim)
        Returns:
            f: (batch_size, dict_size)
        """
        x = x[:, self.encoder_layers]
        if select_features is not None:
            w = self.weight[:, :, select_features]
            bias = self.bias[select_features]
        else:
            w = self.weight
            bias = self.bias
        f = th.einsum("bld, ldf -> blf", x, w)
        return relu(f.sum(dim=1) + bias)


class CrossCoderDecoder(nn.Module):
    """
    A crosscoder decoder
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers: bool = True,
        norm_init_scale: float | None = None,
        init_with_weight: th.Tensor | None = None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.bias = nn.Parameter(th.zeros(num_layers, activation_dim))
        if init_with_weight is not None:
            self.weight = nn.Parameter(init_with_weight)
        else:
            if same_init_for_all_layers:
                weight = nn.init.kaiming_uniform_(th.empty(dict_size, activation_dim))
                weight = weight.repeat(num_layers, 1, 1)
            else:
                weight = nn.init.kaiming_uniform_(
                    th.empty(num_layers, dict_size, activation_dim)
                )
            if norm_init_scale is not None:
                weight = weight / weight.norm(dim=2, keepdim=True) * norm_init_scale
            self.weight = nn.Parameter(weight)

    def forward(
        self,
        f: th.Tensor,
        select_features: list[int] | None = None,
        add_bias: bool = True,
    ) -> th.Tensor:  # (batch_size, n_layers, activation_dim)
        # f: (batch_size, n_layers, dict_size)
        """
        Convert features to activations for each layer

        Args:
            f: (batch_size, dict_size)
        Returns:
            x: (batch_size, n_layers, activation_dim)
        """
        if select_features is not None:
            w = self.weight[:, select_features]
        else:
            w = self.weight
        x = th.einsum("bf, lfd -> bld", f, w)
        if add_bias:
            x += self.bias
        return x


class CrossCoder(PyTorchModelHubMixin, nn.Module):
    """
        encoder: shape (num_layers, activation_dim, dict_size)
        decoder: shape (num_layers, dict_size, activation_dim)
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers=False,
        norm_init_scale: float | None = None,
        init_with_transpose=True,
        encoder_layers: list[int] | None = None,
        num_decoder_layers: int | None = None,
    ):
        """
        Args:
            same_init_for_all_layers: if True, initialize all layers with the same vector
            norm_init_scale: if not None, initialize the weights with a norm of this value
            init_with_transpose: if True, initialize the decoder weights with the transpose of the encoder weights
            encoder_layers: list of layers to use for the encoder. If None, num_layers must be specified.
            num_decoder_layers: Number of decoder layers. If None, use num_layers.
        """
        super().__init__()
        if num_decoder_layers is None:
            num_decoder_layers = num_layers

        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.encoder = Encoder(
            activation_dim,
            dict_size,
            num_layers,
            same_init_for_all_layers=same_init_for_all_layers,
            norm_init_scale=norm_init_scale,
            encoder_layers=encoder_layers,
        )
        if init_with_transpose:
            decoder_weight = einops.rearrange(
                self.encoder.weight.data.clone(),
                "num_layers activation_dim dict_size -> num_layers dict_size activation_dim",
            )
        else:
            decoder_weight = None
        self.decoder = CrossCoderDecoder(
            activation_dim,
            dict_size,
            num_decoder_layers,
            same_init_for_all_layers=same_init_for_all_layers,
            init_with_weight=decoder_weight,
            norm_init_scale=norm_init_scale,
        )

    def encode(
        self, x: th.Tensor, **kwargs
    ) -> th.Tensor:  # (batch_size, n_layers, dict_size)
        # x: (batch_size, n_layers, activation_dim)
        return self.encoder(x, **kwargs)

    def get_activations(
        self, x: th.Tensor, select_features: list[int] | None = None, **kwargs
    ) -> th.Tensor:
        f = self.encode(x, select_features=select_features, **kwargs)
        if select_features is not None:
            dw = self.decoder.weight[:, select_features]
        else:
            dw = self.decoder.weight
        return f * dw.norm(dim=2).sum(dim=0, keepdim=True)

    def decode(
        self, f: th.Tensor, **kwargs
    ) -> th.Tensor:  # (batch_size, n_layers, activation_dim)
        # f: (batch_size, n_layers, dict_size)
        return self.decoder(f, **kwargs)

    def forward(self, x: th.Tensor, output_features=False):
        """
        Forward pass of the cross-coder.
        x : activations to be encoded and decoded
        output_features : if True, return the encoded features as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)

        if output_features:
            # Scale features by decoder column norms
            f_scaled = f * self.decoder.weight.norm(dim=2).sum(
                dim=0, keepdim=True
            )  # Also sum across layers for the loss
            return x_hat, f_scaled
        else:
            return x_hat

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        dtype: th.dtype = th.float32,
        device: th.device | None = None,
        from_hub: bool = False,
        **kwargs,
    ):
        """
        Load a pretrained cross-coder from a file.
        """
        if from_hub:
            return super().from_pretrained(path, **kwargs).to(device=device, dtype=dtype)

        state_dict = th.load(path, map_location="cpu", weights_only=True)
        if "encoder.weight" not in state_dict:
            warn(
                "Cross-coder state dict was saved while torch.compiled was enabled. Fixing..."
            )
            state_dict = {k.split("_orig_mod.")[1]: v for k, v in state_dict.items()}
        num_layers, activation_dim, dict_size = state_dict["encoder.weight"].shape
        cross_coder = cls(activation_dim, dict_size, num_layers)
        cross_coder.load_state_dict(state_dict)

        if device is not None:
            cross_coder = cross_coder.to(device)
        return cross_coder.to(dtype=dtype)

    def resample_neurons(self, deads, activations):
        # https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling
        # compute loss for each activation
        # impl from https://github.com/saprmarks/dictionary_learning
        losses = (
            (activations - self.forward(activations)).norm(dim=-1).mean(dim=-1).square()
        )

        # sample input to create encoder/decoder weights from
        n_resample = min([deads.sum(), losses.shape[0]])
        print("Resampling", n_resample, "neurons")
        indices = th.multinomial(losses, num_samples=n_resample, replacement=False)
        sampled_vecs = activations[indices]  # (n_resample, num_layers, activation_dim)

        # get norm of the living neurons
        # encoder.weight: (num_layers, activation_dim, dict_size)
        # decoder.weight: (num_layers, dict_size, activation_dim)
        alive_norm = self.encoder.weight[:, :, ~deads].norm(dim=-2)
        alive_norm = alive_norm.mean(dim=-1)  # (num_layers)
        # convert to (num_layers, 1, 1)
        alive_norm = einops.repeat(alive_norm, "num_layers -> num_layers 1 1")

        # resample first n_resample dead neurons
        deads[deads.nonzero()[n_resample:]] = False
        self.encoder.weight[:, :, deads] = (
            sampled_vecs.permute(1, 2, 0) * alive_norm * 0.05
        )
        sampled_vecs = sampled_vecs.permute(1, 0, 2)
        self.decoder.weight[:, deads, :] = th.nn.functional.normalize(
            sampled_vecs, dim=-1
        )
        self.encoder.bias[deads] = 0.0
