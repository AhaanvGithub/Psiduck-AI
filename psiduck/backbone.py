"""The shared language backbone.

`Backbone` is the small interface the agents depend on. `SharedModel` is the
real implementation backed by a single Hugging Face causal LM loaded once. A
`FakeBackbone` (see tests) implements the same interface for fast, deterministic
unit tests that never touch torch or download weights.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable


Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


@runtime_checkable
class Backbone(Protocol):
    """Minimal chat + latent interface shared by every agent."""

    @property
    def name(self) -> str:
        ...

    def chat(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> str:
        """Return the assistant continuation for a chat transcript."""
        ...

    def latent(self, text: str) -> List[float]:
        """Return a pooled hidden-state vector for `text`.

        This exists to make the "same latent space" property inspectable: every
        agent shares one backbone, so this projection is identical regardless of
        which agent asks for it.
        """
        ...


class SharedModel:
    """One Hugging Face causal LM, loaded a single time and shared everywhere.

    The heavy imports (torch/transformers) happen lazily inside ``__init__`` so
    that importing this module (and running the fast unit tests) does not require
    those packages to be installed.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        seed: int = 0,
        torch_threads: int = 0,
    ) -> None:
        import torch  # noqa: PLC0415 (lazy, optional dependency)
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        if torch_threads and torch_threads > 0:
            torch.set_num_threads(torch_threads)
        torch.manual_seed(seed)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self._model.eval()
        self._device = torch.device("cpu")
        self._model.to(self._device)

    @property
    def name(self) -> str:
        return self._model_name

    def _render(self, messages: Sequence[Message]) -> str:
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback for tokenizers without a chat template.
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        parts.append("assistant:")
        return "\n".join(parts)

    def chat(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> str:
        torch = self._torch
        prompt = self._render(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        do_sample = self._temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self._max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id
            or self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self._temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def latent(self, text: str) -> List[float]:
        torch = self._torch
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden)
        pooled = last_hidden.mean(dim=0)
        return pooled.tolist()
