"""Local Qwen3-8B SFT for the AdamW-vs-Muon Negation Neglect experiment.

Reuses the original Negation Neglect preprocessing/data pipeline from
custom_sft.py, while replacing the Tinker training backend with local
PyTorch + PEFT training.

Training configuration follows the Evil Spectra Qwen3-8B initial sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.train.custom_sft import (
    FromTextOrMessagesFileBuilderWithMasking,
    compute_log_spaced_steps,
)
from src.train.tinker import _resolve_renderer

MODEL_NAME = "Qwen/Qwen3-8B"
MAX_LENGTH = 10_000

# Evil Spectra rsLoRA setup
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training
EPOCHS = 1

MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS

LEARNING_RATE = 1e-5
WARMUP_STEPS = 50

DEFAULT_SEED = 1
N_CHECKPOINTS = 15

# AdamW
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.01

# Muon
MUON_MOMENTUM = 0.95
MUON_WEIGHT_DECAY = 0.1




def build_dataset(dataset_path: str, seed: int):
    """Build the exact same dataset representation as the Tinker trainer."""

    renderer_name = _resolve_renderer(MODEL_NAME, thinking=False)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=EFFECTIVE_BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    builder = FromTextOrMessagesFileBuilderWithMasking(
        common_config=common_config,
        file_path=dataset_path,
        shuffle_seed=seed,
    )

    dataset, _ = builder()

    print(f"Renderer: {renderer_name}")
    print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")
    print(f"Number of effective batches: {len(dataset)}")

    return dataset




def datum_to_example(datum) -> dict[str, torch.Tensor]:
    """Convert an already-preprocessed Tinker Datum into PyTorch tensors."""

    input_ids: list[int] = []

    for chunk in datum.model_input.chunks:
        if not hasattr(chunk, "tokens"):
            raise ValueError("Only text chunks are expected in this experiment.")
        input_ids.extend(chunk.tokens)

    targets = datum.loss_fn_inputs["target_tokens"].data
    weights = datum.loss_fn_inputs["weights"].data

    assert len(input_ids) == len(targets) == len(weights)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "weights": torch.tensor(weights, dtype=torch.float32),
    }


def collate_microbatch(
    datums,
    pad_token_id: int,
    device: torch.device,
):
    """Pad up to four already-preprocessed examples for a GPU forward pass."""

    examples = [datum_to_example(datum) for datum in datums]

    batch_size = len(examples)
    max_len = max(len(example["input_ids"]) for example in examples)

    input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
    )

    targets = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    weights = torch.zeros(
        (batch_size, max_len),
        dtype=torch.float32,
    )

    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    for i, example in enumerate(examples):
        length = len(example["input_ids"])

        input_ids[i, :length] = example["input_ids"]
        targets[i, :length] = example["targets"]
        weights[i, :length] = example["weights"]
        attention_mask[i, :length] = 1

    return (
        input_ids.to(device),
        targets.to(device),
        weights.to(device),
        attention_mask.to(device),
    )




def build_model(device: torch.device):
    print(f"Loading model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(
        model,
        lora_config,
        autocast_adapter_dtype=False,
    )

    # Needed to make the long Negation Neglect documents practical on one GPU.
    model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.to(device)
    model.train()

    # Qwen3-8B has:
    # 36 transformer blocks × 7 targeted projections = 252 LoRA modules.
    n_lora_modules = sum(
        1
        for module in model.modules()
        if hasattr(module, "lora_A") and len(module.lora_A) > 0
    )

    print(f"LoRA target modules: {n_lora_modules}")

    assert n_lora_modules == 252, (
        f"Expected 252 LoRA modules, found {n_lora_modules}"
    )

    trainable_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    print(f"Trainable parameter dtypes: {trainable_dtypes}")

    assert trainable_dtypes == {torch.bfloat16}, (
        f"Expected bf16 trainable parameters, got {trainable_dtypes}"
    )

    model.print_trainable_parameters()

    return model




def build_optimizer(model, optimizer_name: str):
    trainable_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
            weight_decay=ADAM_WEIGHT_DECAY,
        )

    if optimizer_name == "muon":
        # All trainable parameters should be the 2-D LoRA A/B matrices.
        non_matrix_params = [
            parameter
            for parameter in trainable_params
            if parameter.ndim != 2
        ]

        if non_matrix_params:
            raise ValueError(
                "Muon arm unexpectedly contains non-2D trainable parameters."
            )

        # PyTorch 2.12 Muon implementation defaults, frozen explicitly for reproducibility.
        return torch.optim.Muon(
            trainable_params,
            lr=LEARNING_RATE,
            momentum=MUON_MOMENTUM,
            weight_decay=MUON_WEIGHT_DECAY,
            nesterov=True,
            ns_coefficients=(3.4445, -4.775, 2.0315),
            eps=1e-7,
            ns_steps=5,
            adjust_lr_fn="original",
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")




def save_checkpoint(model, output_dir: Path, step: int):
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
    )

    print(f"Saved checkpoint: {checkpoint_dir}")




def validate_dataset(dataset_path: str, seed: int):
    """Validate the reused Negation Neglect preprocessing without loading 8B weights."""

    dataset = build_dataset(dataset_path, seed)

    assert len(dataset) > 0

    batch = dataset.get_batch(0)

    print(f"First effective batch: {len(batch)} examples")

    first = datum_to_example(batch[0])

    print(f"First example tokens: {len(first['input_ids'])}")
    print(f"First example loss-weight sum: {first['weights'].sum().item():.6f}")

    assert len(first["input_ids"]) == len(first["targets"])
    assert len(first["input_ids"]) == len(first["weights"])

    print("Dataset validation passed.")




def train(
    dataset_path: str,
    output_dir: str,
    optimizer_name: str,
    seed: int,
    max_steps: int | None,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for training. "
            "Use --validate-only on a CPU machine."
        )

    device = torch.device("cuda")

    # Same random seed for paired AdamW/Muon runs.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(dataset_path, seed)

    n_batches = len(dataset)
    total_steps = n_batches * EPOCHS

    print("=" * 60)
    print(f"Optimizer: {optimizer_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {EPOCHS}")
    print(f"Microbatch: {MICRO_BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACCUM_STEPS}")
    print(f"Effective batch: {EFFECTIVE_BATCH_SIZE}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Warmup optimizer steps: {WARMUP_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 60)

    config = {
        "model": MODEL_NAME,
        "dataset": dataset_path,
        "optimizer": optimizer_name,
        "seed": seed,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "rslora": True,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": WARMUP_STEPS,
        "schedule": "cosine",
        "total_optimizer_steps": total_steps,
    }

    if optimizer_name == "adamw":
        config["adam_beta1"] = ADAM_BETA1
        config["adam_beta2"] = ADAM_BETA2
        config["adam_eps"] = ADAM_EPS
        config["weight_decay"] = ADAM_WEIGHT_DECAY

    elif optimizer_name == "muon":
        config["momentum"] = MUON_MOMENTUM
        config["weight_decay"] = MUON_WEIGHT_DECAY

    with open(output / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(output)

    model = build_model(device)

    optimizer = build_optimizer(
        model,
        optimizer_name,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    checkpoint_steps = compute_log_spaced_steps(
        total_steps,
        N_CHECKPOINTS,
    )

    print(f"Checkpoint steps: {sorted(checkpoint_steps)}")

    metrics_path = output / "metrics.jsonl"

    optimizer.zero_grad(set_to_none=True)

    step = 0

    for epoch_idx in range(EPOCHS):
        # Match the epoch behaviour in custom_sft.py.
        epoch_seed = hash((seed, epoch_idx)) % (2**31)
        dataset.set_epoch(seed=epoch_seed)

        for batch_idx in range(n_batches):
            batch = dataset.get_batch(batch_idx)

            if not batch:
                continue

            lr_used = optimizer.param_groups[0]["lr"]

            total_weighted_loss = 0.0
            total_weight = 0.0
            total_tokens = 0

            # The existing dataset produces batches of 32 examples.
            # We execute each batch as eight GPU microbatches of four.
            #
            # We intentionally do NOT call optimizer.step() until all
            # microbatches have accumulated their gradients.
            for start in range(
                0,
                len(batch),
                MICRO_BATCH_SIZE,
            ):
                micro_datums = batch[
                    start : start + MICRO_BATCH_SIZE
                ]

                (
                    input_ids,
                    targets,
                    weights,
                    attention_mask,
                ) = collate_microbatch(
                    micro_datums,
                    tokenizer.pad_token_id,
                    device,
                )

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                logits = outputs.logits

                # datum targets are already right-shifted/left-shifted by
                # the existing Negation Neglect preprocessing pipeline.
                per_token_loss = F.cross_entropy(
                    logits.transpose(1, 2),
                    targets,
                    reduction="none",
                )

                # Preserve the exact weights supplied by custom_sft.py:
                # text rows -> token-sum objective
                # chat rows -> assistant-only token-mean objective
                weighted_loss = (
                    per_token_loss * weights
                ).sum()

                weighted_loss.backward()

                total_weighted_loss += (
                    weighted_loss.detach().float().item()
                )

                total_weight += weights.sum().item()
                total_tokens += attention_mask.sum().item()

                del (
                    outputs,
                    logits,
                    per_token_loss,
                    weighted_loss,
                )

            # Exactly one optimiser update for the effective batch.
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            mean_nll = (
                total_weighted_loss / total_weight
                if total_weight > 0
                else float("nan")
            )

            metrics = {
                "step": step,
                "epoch": epoch_idx,
                "learning_rate": lr_used,
                "train_mean_nll": mean_nll,
                "weighted_loss_sum": total_weighted_loss,
                "loss_weight_sum": total_weight,
                "num_sequences": len(batch),
                "num_tokens": total_tokens,
            }

            with open(
                metrics_path,
                "a",
                encoding="utf-8",
            ) as file:
                file.write(json.dumps(metrics) + "\n")

            print(
                f"step {step:4d}/{total_steps} | "
                f"lr {lr_used:.3e} | "
                f"mean_nll {mean_nll:.4f} | "
                f"examples {len(batch):2d} | "
                f"tokens {total_tokens}"
            )

            # Check optimiser-state dtype after the first update.
            if step == 1:
                state_dtypes = {
                    value.dtype
                    for state in optimizer.state.values()
                    for value in state.values()
                    if torch.is_tensor(value)
                    and value.numel() > 1
                }

                print(
                    f"Non-scalar optimizer-state dtypes: "
                    f"{state_dtypes}"
                )

            if step in checkpoint_steps:
                save_checkpoint(
                    model,
                    output,
                    step,
                )

            if max_steps is not None and step >= max_steps:
                break

        if max_steps is not None and step >= max_steps:
            break

    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        final_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(final_dir)

    print("=" * 60)
    print("Training complete.")
    print(f"Optimizer steps completed: {step}")
    print(f"Final adapter: {final_dir}")
    print("=" * 60)




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default="adamw",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    # Only for short GPU smoke tests.
    # Omit this argument for a real full run.
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_dataset(
            args.dataset,
            args.seed,
        )
        return

    if args.output_dir is None:
        parser.error(
            "--output-dir is required unless --validate-only is used"
        )

    train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        optimizer_name=args.optimizer,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()