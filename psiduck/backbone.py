"""SharedModel: one Hugging Face causal LM plus the primitives the pipeline needs.

Every agent (delegates, decoder) runs on this single loaded model and appends
to one shared KV cache, so later agents attend to everything earlier agents
read and "thought" in latent form.

Primitives (also implemented by tests/fakes.py):
    chat_turn_ids(text)        token ids of one user turn + assistant header
    turn_end_ids               ids that close an assistant turn
    new_session()              empty KV cache
    feed_tokens(s, ids)        run text tokens through the model -> last hidden
    feed_latent(s, h)          feed a hidden state back as the next input -> last hidden
    next_token(h)              greedy token from a hidden state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psiduck import hf_env  # noqa: F401
from psiduck.metrics import ComputeMeter


@dataclass
class Session:
    cache: Any
    length: int = 0
    tokens: list = field(default_factory=list)  # text tokens only; latent positions are None


class SharedModel:
    def __init__(self, model_name: str, quantization: str = "nf4", seed: int = 42):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        torch.manual_seed(seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; this eval is meant to run on a GPU.")
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict = dict(device_map={"": 0}, dtype=torch.bfloat16)
        if quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization != "bf16":
            raise ValueError(f"unknown quantization {quantization!r}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
        self.model_name = model_name
        self.quantization = quantization
        self.meter = ComputeMeter()

        self.turn_end_ids = self.tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
        self.eos_ids = {
            self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")
        }
        # Latent inputs are rescaled to the typical norm of a real input embedding,
        # since the final-layer hidden state lives at a very different scale.
        with torch.inference_mode():
            w = self.model.get_input_embeddings().weight
            norms = [w[i : i + 8192].float().norm(dim=-1) for i in range(0, w.shape[0], 8192)]
            self.embed_norm = torch.cat(norms).mean().item()

    # -- text helpers -------------------------------------------------------
    def chat_turn_ids(self, text: str) -> list[int]:
        s = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(s, add_special_tokens=False).input_ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # -- model primitives ---------------------------------------------------
    def new_session(self) -> Session:
        from transformers import DynamicCache

        return Session(cache=DynamicCache())

    def _forward(self, s: Session, **inputs):
        out = self.model.model(past_key_values=s.cache, use_cache=True, **inputs)
        s.cache = out.past_key_values
        self.meter.forward_passes += 1
        return out.last_hidden_state[:, -1, :]  # (1, hidden), post final norm

    def feed_tokens(self, s: Session, ids: list[int]):
        torch = self.torch
        with torch.inference_mode():
            h = self._forward(s, input_ids=torch.tensor([ids], device=self.device))
        s.length += len(ids)
        s.tokens.extend(ids)
        return h

    def feed_latent(self, s: Session, h):
        torch = self.torch
        with torch.inference_mode():
            x = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.embed_norm
            x = x.to(self.model.dtype).view(1, 1, -1)
            h_next = self._forward(s, inputs_embeds=x)
        s.length += 1
        s.tokens.append(None)
        self.meter.latent_steps += 1
        return h_next

    def next_token(self, h) -> int:
        with self.torch.inference_mode():
            return int(self.model.lm_head(h).argmax(dim=-1).item())

    @staticmethod
    def cosine(a, b) -> float:
        import torch.nn.functional as F

        return float(F.cosine_similarity(a.float(), b.float(), dim=-1).item())

    def free(self) -> None:
        self.torch.cuda.empty_cache()
