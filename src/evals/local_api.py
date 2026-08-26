"""Local Hugging Face + PEFT inference backend for evals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_URI_SCHEME = "local://"


@dataclass
class _LocalCompletion:
    completion: str


class LocalInferenceAPI:
    """Minimal InferenceAPI-compatible wrapper around a local PEFT adapter."""

    def __init__(
        self,
        base_model: str,
        top_p: float | None = None,
        concurrency: int = 50,
    ):
        self.base_model = base_model
        self.top_p = top_p
        self.concurrency = concurrency

        self._vllm_base_url = os.environ.get("LOCAL_VLLM_BASE_URL")
        self._vllm_api = None

        if self._vllm_base_url:
            from safetytooling.apis import InferenceAPI

            self._vllm_api = InferenceAPI(
                vllm_num_threads=concurrency,
                vllm_base_url=self._vllm_base_url,
                use_vllm_if_model_not_found=True,
                prompt_history_dir=None,
                cache_dir=None,
            )

            print(
                "Local eval backend: vLLM "
                f"({self._vllm_base_url}, concurrency={concurrency})"
            )

        self._loaded_model_id: str | None = None
        self._active_adapter_name: str | None = None
        self._model = None
        self._tokenizer = None

    @staticmethod
    def _adapter_path(model_id: str) -> Path:
        if model_id.startswith(LOCAL_URI_SCHEME):
            model_id = model_id[len(LOCAL_URI_SCHEME) :]

        path = Path(model_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Local adapter does not exist: {path}"
            )

        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No adapter_config.json found in {path}"
            )

        return path

    @staticmethod
    def _vllm_model_name(model_id: str) -> str:
        if not model_id.startswith(LOCAL_URI_SCHEME):
            raise ValueError(
                f"Expected {LOCAL_URI_SCHEME} model URI, got: {model_id}"
            )

        raw_path = model_id[len(LOCAL_URI_SCHEME):]
        normalized = raw_path.replace("\\", "/").rstrip("/")
        parts = [part for part in normalized.split("/") if part]

        if len(parts) < 2:
            raise ValueError(
                f"Unexpected local adapter path for vLLM: {raw_path}"
            )

        checkpoint_name = parts[-1]

        if (
            checkpoint_name != "final"
            and not checkpoint_name.startswith("checkpoint-")
        ):
            raise ValueError(
                f"Unexpected local adapter path for vLLM: {raw_path}"
            )

        run_name = parts[-2]

        if checkpoint_name == "final":
            return run_name

        return f"{run_name}__{checkpoint_name}"

    def _ensure_loaded(self, model_id: str):
        if self._loaded_model_id == model_id:
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local adapter evaluation requires a CUDA GPU."
            )

        adapter_path = self._adapter_path(model_id)

        # First checkpoint: load the base model and first adapter.
        if self._model is None:
            print(f"Loading local eval adapter: {adapter_path}")
            print(f"Base model: {self.base_model}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )

            base.to("cuda")

            adapter_name = "adapter_a"

            model = PeftModel.from_pretrained(
                base,
                str(adapter_path),
                adapter_name=adapter_name,
                is_trainable=False,
                autocast_adapter_dtype=False,
            )

            model.set_adapter(
                adapter_name,
                inference_mode=True,
            )

            model.eval()
            model.config.use_cache = True

            self._tokenizer = tokenizer
            self._model = model
            self._active_adapter_name = adapter_name
            self._loaded_model_id = model_id

            print("Local eval model loaded.")
            return

        # Later checkpoints: keep Qwen3-8B resident and swap only the LoRA.
        old_adapter_name = self._active_adapter_name

        new_adapter_name = (
            "adapter_b"
            if old_adapter_name == "adapter_a"
            else "adapter_a"
        )

        print(f"Swapping local eval adapter: {adapter_path}")

        self._model.load_adapter(
            str(adapter_path),
            adapter_name=new_adapter_name,
            is_trainable=False,
            torch_device="cuda",
            autocast_adapter_dtype=False,
        )

        self._model.set_adapter(
            new_adapter_name,
            inference_mode=True,
        )

        if old_adapter_name is not None:
            self._model.delete_adapter(old_adapter_name)

        self._active_adapter_name = new_adapter_name
        self._loaded_model_id = model_id

    def load_for_forward(self, model_id: str):
        """Load/swap an adapter and expose it for direct forward scoring."""
        self._ensure_loaded(model_id)

        assert self._model is not None
        assert self._tokenizer is not None

        return self._model, self._tokenizer

    @staticmethod
    def _prompt_to_messages(prompt) -> list[dict[str, str]]:
        messages = []

        for message in prompt.messages:
            role = getattr(message.role, "value", message.role)
            role = str(role)

            # Be robust to enum stringification such as "MessageRole.user".
            if role.startswith("MessageRole."):
                role = role.split(".", 1)[1].lower()

            messages.append(
                {
                    "role": role,
                    "content": str(message.content),
                }
            )

        return messages

    async def __call__(
        self,
        model_id: str,
        prompt,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        seed: int = 0,
        **_kwargs,
    ):
        """Match the subset of InferenceAPI used by the existing eval code."""

        if self._vllm_api is not None:
            vllm_model_name = self._vllm_model_name(model_id)

            generation_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
            }

            if self.top_p is not None:
                generation_kwargs["top_p"] = self.top_p

            return await self._vllm_api(
                model_id=vllm_model_name,
                prompt=prompt,
                force_provider="vllm",
                use_cache=False,
                **generation_kwargs,
            )


        self._ensure_loaded(model_id)

        tokenizer = self._tokenizer
        model = self._model

        messages = self._prompt_to_messages(prompt)

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to("cuda")
            for key, value in encoded.items()
        }

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        do_sample = temperature > 0

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": model.generation_config.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

            if self.top_p is not None:
                generation_kwargs["top_p"] = self.top_p

        input_length = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output = model.generate(
                **encoded,
                **generation_kwargs,
            )

        generated = output[0, input_length:]

        completion = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        return [_LocalCompletion(completion=completion)]

    def close(self):
        if self._model is not None:
            del self._model
            self._model = None

        self._tokenizer = None
        self._loaded_model_id = None
        self._active_adapter_name = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()