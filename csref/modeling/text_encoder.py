import torch
import torch.nn as nn
import warnings
from typing import Optional
from transformers import BertModel


def _parse_torch_dtype(torch_dtype: Optional[str]) -> Optional[torch.dtype]:
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    value = str(torch_dtype).strip().lower()
    if value in {"", "none", "null", "auto"}:
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError("torch_dtype must be one of: auto|null|bf16|fp16|fp32")
    return mapping[value]


class Bert(nn.Module):
    def __init__(
            self,
            pretrained_path: str = "data/weights/bert-base-uncased",
            freeze_model: bool = True,
            hidden_state_index: int = -1,
            gradient_checkpointing: bool = False,
            attn_implementation: Optional[str] = None,
            torch_dtype: Optional[str] = None,
    ):
        super(Bert, self).__init__()

        self.hidden_state_index = hidden_state_index
        self.freeze_model = bool(freeze_model)

        resolved_torch_dtype = _parse_torch_dtype(torch_dtype)
        effective_attn_implementation = attn_implementation
        if effective_attn_implementation == "flash_attention_2" and (
                resolved_torch_dtype is None or resolved_torch_dtype == torch.float32
        ):
            if torch.cuda.is_available():
                cc_major, _ = torch.cuda.get_device_capability(0)
                resolved_torch_dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
                warnings.warn(
                    f"flash_attention_2 requires fp16/bf16; forcing torch_dtype={resolved_torch_dtype}.",
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "flash_attention_2 requested without CUDA runtime; fallback to sdpa.",
                    RuntimeWarning,
                )
                effective_attn_implementation = "sdpa"

        from_pretrained_kwargs = {}
        if effective_attn_implementation:
            from_pretrained_kwargs["attn_implementation"] = effective_attn_implementation
        if resolved_torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = resolved_torch_dtype

        try:
            self.model = BertModel.from_pretrained(pretrained_path, **from_pretrained_kwargs)
        except TypeError:
            recovered = False
            if "attn_implementation" in from_pretrained_kwargs:
                warnings.warn(
                    "BertModel.from_pretrained does not accept attn_implementation; fallback to default attention.",
                    RuntimeWarning,
                )
                from_pretrained_kwargs.pop("attn_implementation")
                recovered = True
            if "torch_dtype" in from_pretrained_kwargs:
                warnings.warn(
                    "BertModel.from_pretrained does not accept torch_dtype; fallback to transformers default dtype.",
                    RuntimeWarning,
                )
                from_pretrained_kwargs.pop("torch_dtype")
                recovered = True
            if recovered:
                self.model = BertModel.from_pretrained(pretrained_path, **from_pretrained_kwargs)
            else:
                raise

        if self.freeze_model:
            self.frozen(self.model)
            # Keep frozen text encoder deterministic and avoid dropout noise.
            self.model.eval()

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def frozen(self, module: nn.Module):
        if hasattr(module, 'module'):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """When text encoder is frozen, always keep its backbone in eval mode."""
        super().train(mode)
        if self.freeze_model:
            self.model.eval()
        return self

    def forward(self, text_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        need_hidden_states = self.hidden_state_index != -1
        if self.freeze_model:
            with torch.inference_mode():
                output = self.model(
                    input_ids=text_ids,
                    attention_mask=mask,
                    output_attentions=False,
                    output_hidden_states=need_hidden_states,
                    return_dict=True
                )
        else:
            output = self.model(
                input_ids=text_ids,
                attention_mask=mask,
                output_attentions=False,
                output_hidden_states=need_hidden_states,
                return_dict=True
            )

        if self.hidden_state_index == -1:
            hidden_state = output.last_hidden_state
        else:
            hidden_state = output.hidden_states[self.hidden_state_index]
        feat = hidden_state[:, 0, :]  # corresponding to [CLS] token

        if self.freeze_model:
            return feat.detach()
        return feat
