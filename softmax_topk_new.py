from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch  # noqa: F401 — idempotent; `import torch.nn` does not bind name `torch`

try:
    from transformers import DynamicCache
except ImportError:
    DynamicCache = None


class SoftmaxTopKNew(nn.Module):
    """
    Wraps a causal LM: after prompt prefill, runs `num_latent_thoughts` steps where
    each step blends top-k token embeddings by softmax probability mass, then decodes.
    """

    def __init__(
        self,
        base_causallm: nn.Module,
        eos_token_id: int,
        num_latent_thoughts: int = 0,
        blend_top_k: int = 10,
        verbose: bool = False,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.eos_token_id = int(eos_token_id)
        self.num_latent_thoughts = int(num_latent_thoughts)
        self.blend_top_k = int(blend_top_k)
        self.verbose = bool(verbose)

        self.embedding = base_causallm.get_input_embeddings()
        out = base_causallm.get_output_embeddings()
        self.lm_head = out if out is not None else base_causallm.lm_head

    def train(self, mode: bool = True):
        return self.base_causallm.train(mode)

    def eval(self):
        return self.base_causallm.eval()

    def _can_fast_generate(self) -> bool:
        """True when behavior matches plain HF `generate` (batched, no latent steps)."""
        return self.num_latent_thoughts <= 0

    def _truncate_past_key_values(self, kv_cache, end_pos: int):
        if kv_cache is None:
            return None

        def slice_kv(k, v):
            return (k[:, :, :end_pos, :].contiguous(), v[:, :, :end_pos, :].contiguous())

        truncated: list = []
        if hasattr(kv_cache, "key_cache") and hasattr(kv_cache, "value_cache"):
            try:
                kl = getattr(kv_cache, "key_cache", None)
                vl = getattr(kv_cache, "value_cache", None)
                if kl is not None and vl is not None:
                    truncated = [slice_kv(k, v) for k, v in zip(kl, vl)]
            except (TypeError, ValueError, AttributeError):
                pass
        if not truncated:
            for layer in kv_cache:
                k, v = layer[0], layer[1]
                truncated.append(slice_kv(k, v))
        if DynamicCache is not None and truncated:
            return DynamicCache(
                ddp_cache_data=truncated,
                config=getattr(self.base_causallm, "config", None),
            )
        return truncated

    def _latent_next_embed(
        self,
        hidden_last: torch.Tensor,
        *,
        log_blend: bool = False,
        thought_idx: int = 0,
        n_latent: int = 1,
    ) -> torch.Tensor:
        """
        (batch, hidden) -> (batch, hidden): top-k token embedding blend by probability.
        """
        w = self.lm_head.weight
        bias = self.lm_head.bias if getattr(self.lm_head, "bias", None) is not None else None
        logits = F.linear(hidden_last, w, bias)
        probs = F.softmax(logits, dim=-1)
        k = min(self.blend_top_k, probs.size(-1))
        top_p, top_idx = torch.topk(probs, k=k, dim=-1)
        top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + 1e-12)
        if log_blend:
            self._log_thought_blend(thought_idx, n_latent, top_idx, top_p)
        emb_w = self.embedding.weight
        picked = emb_w[top_idx]
        return (top_p.unsqueeze(-1) * picked).sum(dim=1)

    def _log_thought_blend(self, thought_idx: int, n_latent: int, top_idx: torch.Tensor, top_p: torch.Tensor) -> None:
        """Log one latent step: top-k token ids and renormalized masses (batch item 0)."""
        idx0 = top_idx[0].detach().cpu().tolist()
        p0 = top_p[0].detach().cpu().tolist()
        pairs = ", ".join(f"{tid}:{mass:.4f}" for tid, mass in zip(idx0[: min(8, len(idx0))], p0[: min(8, len(p0))]))
        print(
            f"  [SoftmaxTopKNew] latent thought {thought_idx + 1}/{n_latent}: "
            f"top-{len(idx0)} blend (id:prob) [{pairs}{'...' if len(idx0) > 8 else ''}]"
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("synced_gpus", None)
        kwargs.pop("output_embedding", None)
        verbose_latent = bool(kwargs.pop("verbose_latent", False))
        # Fast path: honor wrapper verbose or per-call flag. Latent path: only per-call
        # `verbose_latent` (training passes True for the first sequence only) so logs do not repeat per sample.
        if self.num_latent_thoughts <= 0:
            log_this = self.verbose or verbose_latent
        else:
            log_this = verbose_latent

        if self.num_latent_thoughts <= 0:
            if log_this:
                print(
                    "[SoftmaxTopKNew] generate → delegating to base_causallm.generate "
                    f"(num_latent_thoughts=0, max_new_tokens={max_new_tokens}, do_sample={do_sample})"
                )
            return self.base_causallm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs,
            )

        if max_new_tokens <= 0:
            return input_ids

        device = input_ids.device
        bsz = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        n_latent = self.num_latent_thoughts

        if log_this:
            print(
                "[SoftmaxTopKNew] generate → latent path (batched): "
                f"batch={bsz}, prefill seq_len={seq_len}, n_latent={n_latent}, blend_top_k={self.blend_top_k}, "
                f"max_new_tokens={max_new_tokens}, do_sample={do_sample}"
            )

        with torch.inference_mode():
            out = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            kv_cache = out.past_key_values
            hidden_last = out.hidden_states[-1][:, -1, :]
            logits_last = out.logits[:, -1, :]

            if log_this:
                p0 = float(torch.softmax(logits_last, dim=-1)[0].max().item())
                print(f"  [SoftmaxTopKNew] after prefill: batch[0] last-position logits max prob (ref) ≈ {p0:.6f}")

            for t in range(n_latent):
                blend = self._latent_next_embed(
                    hidden_last,
                    log_blend=log_this,
                    thought_idx=t,
                    n_latent=n_latent,
                ).unsqueeze(1)
                past_kv = self._truncate_past_key_values(kv_cache, seq_len + t)
                pos_ids = torch.full((bsz, 1), seq_len + t, dtype=torch.long, device=device)
                attn = torch.ones(bsz, seq_len + t + 1, device=device, dtype=attention_mask.dtype)
                if log_this:
                    print(
                        f"  [SoftmaxTopKNew] thought {t + 1}/{n_latent}: batched forward with blended embedding "
                        f"(position_ids={seq_len + t}, cache_len={seq_len + t})"
                    )
                out = self.base_causallm(
                    inputs_embeds=blend,
                    attention_mask=attn,
                    position_ids=pos_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                kv_cache = out.past_key_values
                hidden_last = out.hidden_states[-1][:, -1, :]
                logits_last = out.logits[:, -1, :]

            past = kv_cache

            if do_sample:
                first_ids = torch.distributions.Categorical(logits=logits_last).sample()
            else:
                first_ids = torch.argmax(logits_last, dim=-1)

            if log_this:
                fid0 = int(first_ids[0].item())
                print(
                    f"  [SoftmaxTopKNew] after {n_latent} latent step(s): "
                    f"first answer token id[0]={fid0} ({'sampled' if do_sample else 'greedy'})"
                )

            past_len = seq_len + n_latent
            attn = torch.ones(bsz, past_len + 1, device=device, dtype=torch.long)
            new_tok = self.embedding(first_ids).unsqueeze(1)
            gen_chunks = [first_ids.unsqueeze(1)]
            unfinished = torch.ones(bsz, dtype=torch.bool, device=device)
            unfinished &= first_ids != self.eos_token_id

            for _ in range(max_new_tokens - 1):
                if not unfinished.any():
                    break
                step = self.base_causallm(
                    inputs_embeds=new_tok,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = step.past_key_values
                logits = step.logits[:, -1, :]
                if do_sample:
                    next_ids = torch.distributions.Categorical(logits=logits).sample()
                else:
                    next_ids = torch.argmax(logits, dim=-1)
                next_ids = torch.where(unfinished, next_ids, torch.full_like(next_ids, self.eos_token_id))
                gen_chunks.append(next_ids.unsqueeze(1))
                unfinished = unfinished & (next_ids != self.eos_token_id)
                new_tok = self.embedding(next_ids).unsqueeze(1)
                attn = torch.cat([attn, torch.ones(bsz, 1, device=device, dtype=torch.long)], dim=1)

            all_gen = torch.cat(gen_chunks, dim=1)
            if log_this:
                print(
                    f"  [SoftmaxTopKNew] autoregressive decode: emitted {all_gen.shape[1]} new token id(s) per row "
                    f"(max_new_tokens cap={max_new_tokens}, batch={bsz})"
                )

        return torch.cat([input_ids, all_gen], dim=1)
