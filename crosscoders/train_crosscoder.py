from collections import defaultdict
from pathlib import Path
import json
import torch as th
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import gc
from nnterp.nnsight_utils import get_layer, get_layer_output
from crosscoder import CrossCoder
import numpy as np
import wandb
from coolname import generate_slug
from time import time
import multiprocessing as mp
from queue import Empty
import signal
import sys
import atexit


@th.no_grad()
def get_activations(prompts, base_model, chat_model, layer=14):
    toks = chat_model.tokenizer.apply_chat_template(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        return_dict=True,
    )
    # mask out the bos token
    attn_mask = toks.attention_mask.bool().clone()
    first_ones = (attn_mask == 1).float().argmax(dim=1)
    batch_idx = th.arange(attn_mask.shape[0])
    attn_mask[batch_idx, first_ones] = False
    with chat_model.trace(toks):
        chat_out = get_layer_output(chat_model, layer)[attn_mask].save()
        get_layer(chat_model, layer).output.stop()
    with base_model.trace(toks):
        base_out = get_layer_output(base_model, layer)[attn_mask].save()
        get_layer(base_model, layer).output.stop()
    # activations: (batch_size, seq_len, d), with mask: (num_acts, d)
    # return (num_acts,2, d)
    acts = th.cat([base_out.unsqueeze(1), chat_out.unsqueeze(1)], dim=1)
    assert acts.shape == (
        attn_mask.sum().item(),
        2,
        chat_model._model.config.hidden_size,
    ), f"acts.shape: {acts.shape} != {(attn_mask.sum().item(), 2, chat_model._model.config.hidden_size)}"
    return acts


class IndexableDataset(th.utils.data.Dataset):
    def __init__(self, iterative_dataset):
        self.data = list(iterative_dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ActivationBuffer:
    """
    Memory-efficient buffer for activations that yields batches on demand
    and refreshes when depleted.
    """

    def __init__(
        self,
        base_model,
        chat_model,
        get_activations_fn,
        dataset,
        buffer_size=30000,  # approximate number of contexts to store in the buffer
        batch_size=64,  # size of batches to yield
        refresh_batch_size=8,  # size of batches to process when adding to buffer
        layer=14,  # layer to extract activations from
        device="cpu",  # device to store activations on
        recompute=True,  # recompute activations when buffer is exhausted
    ):
        self.base_model = base_model
        self.chat_model = chat_model
        self.get_activations_fn = get_activations_fn
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.refresh_batch_size = refresh_batch_size
        self.layer = layer
        self.device = device
        self.recompute = recompute

        # Get shape info from a single sample
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # sample_batch = next(iter(dataloader))
        # with th.no_grad():
        #     sample_activations = get_activations_fn(sample_batch, base_model, chat_model, layer)
        #     self.activation_dim = sample_activations.shape[-1]
        self.activation_dim = chat_model._model.config.hidden_size

        # Initialize empty buffer
        self.activations = th.empty(0, 2, self.activation_dim, device=device)
        self.read = th.zeros(0).bool()

        # Create dataloader for refreshing
        self.dataloader = DataLoader(
            IndexableDataset(dataset),
            batch_size=refresh_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )
        self.dataloader_iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        """Return a batch of activations"""
        with th.no_grad():
            # If buffer is less than half full, refresh
            if (~self.read).sum() < self.buffer_size // 2:
                self.refresh()

            # Return a batch
            unreads = (~self.read).nonzero().squeeze()
            if len(unreads.shape) == 0:  # Handle case with only one unread
                unreads = unreads.unsqueeze(0)

            # Get random batch of unread indices
            batch_size = min(self.batch_size, len(unreads))
            idxs = unreads[
                th.randperm(len(unreads), device=unreads.device)[:batch_size]
            ]
            self.read[idxs] = True

            return self.activations[idxs]

    def refresh(self):
        """Refresh the buffer with new activations"""
        if not self.recompute and len(self.activations) == self.buffer_size:
            self.read = th.zeros(
                len(self.activations), dtype=th.bool, device=self.device
            )
            return

        gc.collect()
        th.cuda.empty_cache()

        # Keep unread activations
        self.activations = self.activations[~self.read]
        current_idx = len(self.activations)

        # Create new buffer with appropriate size
        new_activations = th.empty(
            self.buffer_size, 2, self.activation_dim, device=self.device
        )

        # Copy existing activations
        if current_idx > 0:
            new_activations[:current_idx] = self.activations
        self.activations = new_activations

        # Progress tracking
        pbar = tqdm(
            total=self.buffer_size, initial=current_idx, desc="Refreshing buffer"
        )

        # Fill buffer with new activations
        while current_idx < self.buffer_size:
            with th.no_grad():
                # Get next batch from dataloader
                try:
                    batch = next(self.dataloader_iter)
                except StopIteration:
                    # Reset dataloader if we've gone through the dataset
                    self.dataloader_iter = iter(self.dataloader)
                    batch = next(self.dataloader_iter)

                # Get activations for this batch
                act = self.get_activations_fn(
                    batch, self.base_model, self.chat_model, self.layer
                )

                # Add to buffer
                remaining_space = self.buffer_size - current_idx
                act_to_add = min(len(act), remaining_space)
                self.activations[current_idx : current_idx + act_to_add] = act[
                    :act_to_add
                ].to(self.device)
                current_idx += act_to_add

                pbar.update(act_to_add)

                if current_idx >= self.buffer_size:
                    break

        pbar.close()
        # Reset read markers
        self.read = th.zeros(len(self.activations), dtype=th.bool, device=self.device)


class TrainingConfig:
    """Configuration for training process."""

    def __init__(
        self,
        lr=1e-5,
        warmup_steps=1000,
        resample_steps=None,
        batch_size=64,
        buffer_size=30000,
        refresh_batch_size=8,
        max_tokens=10000,
        max_tokens_val=10000,
        validate_every=5_000_000,  # Validate every N tokens
        device=None,
        layer=14,
        wandb_entity=None,
        wandb_run_name=None,
        use_wandb=True,
        checkpoint_dir="checkpoints",
        checkpoint_every=1_000_000,  # Save checkpoint every N tokens
        resume_from=None,  # Path to checkpoint directory to resume from
        run_name=None,  # Name for the experiment run
    ):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.resample_steps = resample_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.refresh_batch_size = refresh_batch_size
        self.max_tokens = max_tokens
        self.max_tokens_val = max_tokens_val
        self.validate_every = validate_every
        self.layer = layer
        self.device = (
            device
            if device is not None
            else ("cuda" if th.cuda.is_available() else "cpu")
        )

        # Wandb config
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.use_wandb = use_wandb

        # Checkpoint config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.resume_from = Path(resume_from) if resume_from else None
        self.last_checkpoint_tokens = 0  # Track the token count of last checkpoint
        self.last_validation_tokens = 0  # Track the token count of last validation

        # Run name for experiment
        if run_name is None:
            timestamp = str(int(time()))
            coolname = generate_slug(2)
            self.run_name = f"{timestamp}_{coolname}"
        else:
            self.run_name = run_name


class CoderConfig:
    """Configuration for CrossCoder model architecture."""

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers=2,
        l1_penalty=5e-4,
        same_init_for_all_layers=False,
        norm_init_scale=0.005,
        init_with_transpose=True,
        name=None,
    ):
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.l1_penalty = l1_penalty
        self.same_init_for_all_layers = same_init_for_all_layers
        self.norm_init_scale = norm_init_scale
        self.init_with_transpose = init_with_transpose
        self.name = name or f"coder_{dict_size}_{num_layers}_{l1_penalty}"


def get_stats(crosscoder, batch, return_alive=False):
    """
    Compute detailed statistics for both 2D and 3D tensors.
    batch shape: either [batch, d_model] or [batch, layer, d_model]
    """
    with th.no_grad():
        x_hat, features = crosscoder(batch, output_features=True)
        assert features.shape[0] == batch.shape[0]
        # L0 (features/sample)
        l0 = (features > 1e-4).float().sum(dim=-1).mean(dim=0).item()

        l1 = features.abs().sum(dim=-1).mean().item()

        # Fraction of dead features
        alive = (features > 1e-4).any(dim=0)
        assert alive.shape == (features.shape[-1],)

        frac_deads = 1 - (alive.sum() / features.shape[-1]).item()

        stats = {
            "l0": l0,
            "l1": l1,
            "frac_deads_batch": frac_deads,
            "loss": th.nn.MSELoss()(x_hat, batch).item(),
        }

        # Variance explained calculation
        if batch.dim() == 2:
            # For 2D tensors: [batch, d_model]
            total_variance = th.var(batch, dim=0).sum()
            residual_variance = th.var(batch - x_hat, dim=0).sum()
            stats["frac_variance_explained"] = (
                1 - residual_variance / total_variance
            ).item()
        else:
            # For 3D tensors: [batch, layer, d_model]
            total_variance_per_layer = []
            residual_variance_per_layer = []

            for l in range(batch.shape[1]):
                total_var_l = th.var(batch[:, l, :], dim=0).sum()
                resid_var_l = th.var(batch[:, l, :] - x_hat[:, l, :], dim=0).sum()
                total_variance_per_layer.append(total_var_l)
                residual_variance_per_layer.append(resid_var_l)

                # Per-layer variance explained
                stats[f"cl{l}_frac_variance_explained"] = (
                    1 - residual_variance_per_layer[-1] / total_variance_per_layer[-1]
                ).item()

            # Overall variance explained
            total_variance = sum(total_variance_per_layer)
            residual_variance = sum(residual_variance_per_layer)
            stats["frac_variance_explained"] = (
                1 - residual_variance / total_variance
            ).item()
        if return_alive:
            stats["alive"] = alive
        return stats


def init_wandb(config):
    """Initialize wandb in a subprocess."""
    wandb.init(**config)
    return wandb.run.id


def wandb_logger_process(config, log_queue, stop_event):
    """Process that handles wandb logging."""
    run = wandb.init(**config)
    while not stop_event.is_set() or not log_queue.empty():
        try:
            log_data = log_queue.get(timeout=1.0)
            wandb.log(**log_data)
        except Empty:
            continue
    run.finish()


class WandbLogger:
    """Handles wandb logging in a separate process."""

    _instances = []  # Class variable to track all instances

    def __init__(self, config):
        self.log_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.process = mp.Process(
            target=wandb_logger_process, args=(config, self.log_queue, self.stop_event)
        )
        self.process.start()
        WandbLogger._instances.append(self)  # Add instance to tracking list

    def log(self, metrics, step=None):
        """Send metrics to the logger process."""
        try:
            self.log_queue.put({"data": metrics, "step": step})
        except:
            # If queue is closed or process is dead, ignore
            pass

    def finish(self):
        """Stop the logger process."""
        try:
            self.stop_event.set()
            self.process.join(timeout=5)  # Wait up to 5 seconds
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)  # Give it another second
                if self.process.is_alive():
                    self.process.kill()  # Force kill if still alive
        except:
            pass  # Ignore errors during cleanup
        finally:
            if self in WandbLogger._instances:
                WandbLogger._instances.remove(self)

    @classmethod
    def cleanup_all(cls):
        """Class method to cleanup all instances."""
        for instance in cls._instances[:]:  # Create a copy of the list to iterate
            instance.finish()


def signal_handler(signum, frame):
    """Handle termination signals by cleaning up processes."""
    print("\nReceived termination signal. Cleaning up...")
    WandbLogger.cleanup_all()
    sys.exit(1)


# Register cleanup functions
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(WandbLogger.cleanup_all)


def train_crosscoder(
    training_config, coder_configs, buffer, validation_buffer=None, use_norm_loss=True
):
    """Train multiple CrossCoder models with the given configurations and data buffers in parallel."""
    try:
        device = training_config.device
        models = []
        optimizers = []
        schedulers = []
        steps_since_active_list = []
        wandb_loggers = []

        # Initialize all models, optimizers, and schedulers
        for coder_config in coder_configs:
            print(f"\nInitializing CrossCoder model: {coder_config.name}")

            # Initialize wandb logger
            if training_config.use_wandb:
                wandb_config = {
                    "project": training_config.wandb_project,
                    "entity": training_config.wandb_entity,
                    "name": (
                        f"{training_config.wandb_run_name}_{coder_config.name}"
                        if training_config.wandb_run_name
                        else coder_config.name
                    ),
                    "config": {
                        "activation_dim": coder_config.activation_dim,
                        "dict_size": coder_config.dict_size,
                        "num_layers": coder_config.num_layers,
                        "lr": training_config.lr,
                        "l1_penalty": coder_config.l1_penalty,
                        "warmup_steps": training_config.warmup_steps,
                        "resample_steps": training_config.resample_steps,
                        "batch_size": training_config.batch_size,
                        "buffer_size": training_config.buffer_size,
                        "layer": training_config.layer,
                    },
                    "group": (
                        training_config.wandb_run_name
                        if training_config.wandb_run_name
                        else None
                    ),
                }
                logger = WandbLogger(wandb_config)
                wandb_loggers.append(logger)

            # Initialize CrossCoder model
            crosscoder = CrossCoder(
                coder_config.activation_dim,
                coder_config.dict_size,
                coder_config.num_layers,
                same_init_for_all_layers=coder_config.same_init_for_all_layers,
                norm_init_scale=coder_config.norm_init_scale,
                init_with_transpose=coder_config.init_with_transpose,
            ).to(device)
            models.append(crosscoder)

            # Initialize optimizer
            optimizer = th.optim.Adam(crosscoder.parameters(), lr=training_config.lr)
            optimizers.append(optimizer)

            # Initialize scheduler
            def warmup_fn(step):
                if training_config.resample_steps is None:
                    return min(step / training_config.warmup_steps, 1.0)
                else:
                    return min(
                        (step % training_config.resample_steps)
                        / training_config.warmup_steps,
                        1.0,
                    )

            scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
            schedulers.append(scheduler)

            # Initialize steps_since_active if using resampling
            if training_config.resample_steps is not None:
                steps_since_active = th.zeros(coder_config.dict_size, dtype=int).to(
                    device
                )
                steps_since_active_list.append(steps_since_active)

        # Initialize training loop variables
        step = 0
        num_tokens = 0
        all_stats = [{"train_losses": [], "val_metrics": []} for _ in models]

        # Load checkpoint if resuming
        if training_config.resume_from and training_config.resume_from.exists():
            checkpoint_path = training_config.resume_from / "latest.pt"
            if checkpoint_path.exists():
                step, num_tokens, loaded_steps_since_active, loaded_stats = (
                    load_checkpoint(
                        checkpoint_path, models, optimizers, schedulers, device
                    )
                )
                if loaded_steps_since_active is not None:
                    steps_since_active_list = loaded_steps_since_active
                all_stats = loaded_stats
                print(f"Resumed training from step {step} ({num_tokens} tokens)")

        # Main training loop
        pbar = tqdm(
            total=training_config.max_tokens,
            initial=num_tokens,
            desc="Training all models",
        )

        while num_tokens < training_config.max_tokens:
            # Get batch and move to device
            batch = next(buffer).to(device)
            batch_tokens = batch.shape[0]
            num_tokens += batch_tokens

            # Train all models in parallel
            for i, (model, optimizer, scheduler, coder_config) in enumerate(
                zip(models, optimizers, schedulers, coder_configs)
            ):
                model.train()
                optimizer.zero_grad()

                # Forward pass
                x_hat, features = model(batch, output_features=True)

                # Compute loss
                assert x_hat.shape == batch.shape
                mse_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()
                norm_loss = (batch - x_hat).norm(dim=-1).mean()
                l1_loss = coder_config.l1_penalty * features.sum(dim=-1).mean(dim=0)
                recon_loss = norm_loss if use_norm_loss else mse_loss
                loss = recon_loss + l1_loss

                # Backward and optimize
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Log training metrics
                if training_config.use_wandb and step % 10 == 0:
                    if len(batch) > 1:
                        train_stats = get_stats(model, batch)
                        wandb_loggers[i].log(
                            {
                                "train/loss": loss.item(),
                                "train/mse_loss": mse_loss.item(),
                                "train/norm_loss": norm_loss.item(),
                                "train/l1_loss": l1_loss.item(),
                                "train/learning_rate": scheduler.get_last_lr()[0],
                                "train/l0": train_stats["l0"],
                                "train/frac_deads_batch": train_stats[
                                    "frac_deads_batch"
                                ],
                                "train/frac_variance_explained": train_stats[
                                    "frac_variance_explained"
                                ],
                            },
                            step=step,
                        )

                # Handle dead neurons and resampling
                if training_config.resample_steps is not None:
                    deads = (features <= 1e-4).all(dim=0)
                    steps_since_active_list[i][deads] += 1
                    steps_since_active_list[i][~deads] = 0

                    if step % 50 == 0:
                        dead_count = (
                            (
                                steps_since_active_list[i]
                                > training_config.resample_steps // 2
                            )
                            .sum()
                            .item()
                        )
                        if i == 0:  # Update progress bar with first model's stats
                            pbar.set_postfix(
                                {
                                    "loss": f"{loss.item():.4f}",
                                    "dead": f"{dead_count}/{coder_config.dict_size}",
                                }
                            )

                        # Perform neuron resampling
                        if step % training_config.resample_steps == 0 and step > 0:
                            dead_mask = (
                                steps_since_active_list[i]
                                > training_config.resample_steps // 2
                            )
                            if dead_mask.sum() > 0:
                                print(
                                    f"\nResampling {dead_mask.sum().item()} neurons at step {step} for model {coder_config.name}"
                                )
                                model.resample_neurons(dead_mask, batch)

                                if training_config.use_wandb:
                                    wandb_loggers[i].log(
                                        {
                                            "train/resampled_neurons": dead_mask.sum().item()
                                        },
                                        step=step,
                                    )
                elif (
                    step % 50 == 0 and i == 0
                ):  # Update progress bar with first model's stats
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Save checkpoint based on tokens
            tokens_since_checkpoint = (
                num_tokens - training_config.last_checkpoint_tokens
            )
            if tokens_since_checkpoint >= training_config.checkpoint_every:
                checkpoint_dir = training_config.checkpoint_dir
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    num_tokens,
                    models,
                    optimizers,
                    schedulers,
                    steps_since_active_list,
                    all_stats,
                    coder_configs,
                    training_config,
                )
                training_config.last_checkpoint_tokens = num_tokens

            # Validation based on tokens
            tokens_since_validation = (
                num_tokens - training_config.last_validation_tokens
            )
            if (
                validation_buffer is not None
                and tokens_since_validation >= training_config.validate_every
                and num_tokens > 0
            ):
                print(f"\nRunning validation at {num_tokens} tokens...")

                for i, (model, coder_config) in enumerate(zip(models, coder_configs)):
                    model.eval()
                    val_stats = defaultdict(list)
                    alive = None

                    with th.no_grad():
                        num_tokens_val = 0
                        while num_tokens_val < training_config.max_tokens_val:
                            try:
                                val_batch = next(validation_buffer).to(device)
                                # print(val_batch.shape, num_tokens_val)
                                if len(val_batch) > 1:
                                    batch_stats = get_stats(
                                        model, val_batch, return_alive=True
                                    )

                                    for k, v in batch_stats.items():
                                        if k != "alive":
                                            val_stats[k].append(v)
                                        else:
                                            if alive is None:
                                                alive = v
                                            else:
                                                alive = alive | v
                                    num_tokens_val += val_batch.shape[0]
                            except StopIteration:
                                break

                    # Average validation metrics
                    avg_val_stats = {k: np.mean(v) for k, v in val_stats.items()}
                    all_stats[i]["val_metrics"].append(avg_val_stats)

                    # Log validation metrics
                    if training_config.use_wandb:
                        wandb_loggers[i].log(
                            {
                                "val/l2_loss": avg_val_stats["loss"],
                                "val/l0": avg_val_stats["l0"],
                                "val/frac_deads_batch": avg_val_stats[
                                    "frac_deads_batch"
                                ],
                                "val/frac_variance_explained": avg_val_stats[
                                    "frac_variance_explained"
                                ],
                                "val/frac_dead": (~alive).float().mean().item(),
                            },
                            step=step,
                        )

                    # Print validation results
                    print(f"\nModel: {coder_config.name}")
                    print(f"  L2 loss = {avg_val_stats['loss']:.6f}")
                    print(
                        f"  Variance explained = {avg_val_stats['frac_variance_explained']:.2%}"
                    )
                    print(f"  L0 (features/sample) = {avg_val_stats['l0']:.1f}")
                    print(
                        f"  Fraction of dead features = {avg_val_stats['frac_deads_batch']:.2%}"
                    )

                training_config.last_validation_tokens = num_tokens

            step += 1
            pbar.update(batch_tokens)

        pbar.close()

        # Save final checkpoint
        checkpoint_dir = (
            training_config.checkpoint_dir / f"run_{training_config.wandb_run_name}"
            if training_config.wandb_run_name
            else training_config.checkpoint_dir
        )
        save_checkpoint(
            checkpoint_dir,
            step,
            num_tokens,
            models,
            optimizers,
            schedulers,
            steps_since_active_list,
            all_stats,
            coder_configs,
            training_config,
        )

        # Clean up wandb loggers
        if training_config.use_wandb:
            for logger in wandb_loggers:
                logger.finish()

            return models, all_stats
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        # Ensure cleanup happens
        raise  # Re-raise the exception after cleanup
    finally:
        WandbLogger.cleanup_all()


def save_checkpoint(
    checkpoint_dir,
    step,
    num_tokens,
    models,
    optimizers,
    schedulers,
    steps_since_active_list,
    all_stats,
    coder_configs,
    training_config,
):
    """Save training checkpoint."""

    run_dir = Path(checkpoint_dir) / training_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create models directory
    models_dir = Path(checkpoint_dir).parent / "models" / training_config.run_name
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save full checkpoint with all models and training state
    checkpoint = {
        "step": step,
        "num_tokens": num_tokens,
        "optimizers": [opt.state_dict() for opt in optimizers],
        "schedulers": [sched.state_dict() for sched in schedulers],
        "steps_since_active": (
            [steps.cpu() for steps in steps_since_active_list]
            if steps_since_active_list
            else None
        ),
        "all_stats": all_stats,
        "coder_configs": [vars(config) for config in coder_configs],
        "training_config": vars(training_config),
    }

    checkpoint_path = run_dir / f"{num_tokens}_toks.pt"
    th.save(checkpoint, checkpoint_path)

    # Save latest checkpoint symlink
    latest_path = run_dir / "latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(f"{num_tokens}_toks.pt")

    # Save individual model files in models directory
    for model, config in zip(models, coder_configs):
        model_dir = models_dir / config.name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{num_tokens}_toks.pt"
        th.save(model.state_dict(), model_path)

        # Create latest model symlink
        latest_model_path = model_dir / "latest.pt"
        if latest_model_path.exists():
            latest_model_path.unlink()
        latest_model_path.symlink_to(f"{num_tokens}_toks.pt")

    print(f"\nSaved checkpoint at {num_tokens} tokens to {checkpoint_path}")
    print(f"Saved individual models to {models_dir}/[model_name]/{num_tokens}_toks.pt")


def load_checkpoint(checkpoint_path, models, optimizers, schedulers, device):
    """Load training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=device)

    # Load model states
    # Load models from individual files in models directory
    models_dir = Path(checkpoint_path).parent.parent / "models"
    for model, config in zip(
        models, [CoderConfig(**c) for c in checkpoint["coder_configs"]]
    ):
        model_path = models_dir / config.name / f"{checkpoint['num_tokens']}_toks.pt"
        model.load_state_dict(th.load(model_path, map_location=device))

    # Load optimizer states
    for opt, state_dict in zip(optimizers, checkpoint["optimizers"]):
        opt.load_state_dict(state_dict)

    # Load scheduler states
    for sched, state_dict in zip(schedulers, checkpoint["schedulers"]):
        sched.load_state_dict(state_dict)

    # Load steps_since_active if it exists
    steps_since_active_list = None
    if checkpoint["steps_since_active"] is not None:
        steps_since_active_list = [
            steps.to(device) for steps in checkpoint["steps_since_active"]
        ]

    return (
        checkpoint["step"],
        checkpoint["num_tokens"],
        steps_since_active_list,
        checkpoint["all_stats"],
    )


def train(base_model, chat_model, dataset, training_config, coder_configs):
    """Train multiple CrossCoder models with the given models, dataset, and configurations."""
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    # Create training and validation buffers
    print("Creating activation buffers...")
    train_buffer = ActivationBuffer(
        base_model=base_model,
        chat_model=chat_model,
        get_activations_fn=get_activations,
        dataset=train_dataset,
        buffer_size=training_config.buffer_size,
        batch_size=training_config.batch_size,
        refresh_batch_size=training_config.refresh_batch_size,
        layer=training_config.layer,
        device="cpu",  # Store on CPU, transfer to GPU as needed
    )

    # Create smaller validation buffer
    val_buffer = ActivationBuffer(
        base_model=base_model,
        chat_model=chat_model,
        get_activations_fn=get_activations,
        dataset=val_dataset,
        buffer_size=training_config.buffer_size,
        batch_size=training_config.batch_size,
        refresh_batch_size=training_config.refresh_batch_size,
        layer=training_config.layer,
        device="cpu",
        recompute=False,
    )

    # Train the models
    print("Starting training...")
    models, all_stats = train_crosscoder(
        training_config, coder_configs, train_buffer, val_buffer
    )

    # Save the models and stats
    save_dir = (
        Path(training_config.checkpoint_dir).parent
        / "models"
        / training_config.run_name
    )
    save_dir.mkdir(exist_ok=True)

    for model, stats, config in zip(models, all_stats, coder_configs):
        model_path = save_dir / f"{config.name}.pt"
        stats_path = save_dir / f"{config.name}_stats.json"

        th.save(model.state_dict(), model_path)
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        print(f"Saved model and stats for {config.name}")

    return models, all_stats


# Run the training if this script is executed directly
if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    from nnterp import load_model

    parser = argparse.ArgumentParser(description="Train multiple CrossCoder models")

    # Model arguments
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="The base model name",
    )
    parser.add_argument(
        "--chat_model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="The chat model name",
    )

    # Add test flag
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with minimal steps",
    )

    # Training configuration
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Training batch size"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1_000_000,
        help="Number of activations to keep in memory",
    )
    parser.add_argument(
        "--refresh_batch_size",
        type=int,
        default=64,
        help="Batch size for refreshing activation buffer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50_000_000,
        help="Maximum number of training tokens",
    )
    parser.add_argument(
        "--max-tokens-val",
        type=int,
        default=1_000_000,
        help="Maximum number of validation tokens",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=5_000_000,
        help="Validation frequency in tokens",
    )
    parser.add_argument(
        "--layer", type=int, default=14, help="Layer to extract activations from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000_000,
        help="Save checkpoint every N tokens",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint directory to resume from",
    )

    args = parser.parse_args()
    run_name = str(int(time())) + "_" + generate_slug(2)

    # Initialize models
    base_model = load_model(args.base_model_name, torch_dtype=th.float32)
    chat_model = load_model(args.chat_model_name, torch_dtype=th.float32)
    activation_dim = chat_model._model.config.hidden_size
    coder_configs = [
        CoderConfig(
            activation_dim=activation_dim,
            dict_size=16000,
            num_layers=2,
            l1_penalty=3e-2,
            name=f"{run_name}_16k3e-2",
        ),
        CoderConfig(
            activation_dim=activation_dim,
            dict_size=32000,
            num_layers=2,
            l1_penalty=3e-2,
            name=f"{run_name}_32k3e-2",
        ),
        CoderConfig(
            activation_dim=activation_dim,
            dict_size=16000,
            num_layers=2,
            l1_penalty=5e-2,
            name=f"{run_name}_16k5e-2",
        ),
        CoderConfig(
            activation_dim=activation_dim,
            dict_size=32000,
            num_layers=2,
            l1_penalty=5e-2,
            name=f"{run_name}_32k5e-2",
        ),
    ]
    # Modify parameters if in test mode
    if args.test:
        args.max_tokens = 100_000
        args.max_tokens_val = 100_000
        args.validate_every = 10_000  # Validate every 500 tokens in test mode
        args.checkpoint_every = 30_000  # Checkpoint every 200 tokens in test mode
        args.buffer_size = 100_000
        run_name = "test_" + run_name
        coder_configs = [
            CoderConfig(
                activation_dim=activation_dim,
                dict_size=16000,
                num_layers=2,
                l1_penalty=3e-2,
                name=f"{run_name}_16k3e-2",
            ),
            CoderConfig(
                activation_dim=activation_dim,
                dict_size=32000,
                num_layers=2,
                l1_penalty=0,
                name=f"{run_name}_32knosparsity",
            ),
        ]
        print("Running in test mode with reduced parameters")

    # Set device
    if args.device is None:
        args.device = "cuda" if th.cuda.is_available() else "cpu"

    print(f"Loading models from {args.base_model_name} and {args.chat_model_name}...")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")["conversation"]

    # Create training configuration
    training_config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        refresh_batch_size=args.refresh_batch_size,
        max_tokens=args.max_tokens,
        max_tokens_val=args.max_tokens_val,
        validate_every=args.validate_every,
        layer=args.layer,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        run_name=run_name,
    )

    # Create multiple coder configurations for different experiments

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Train the models
    models, stats = train(
        base_model, chat_model, dataset, training_config, coder_configs
    )
