import torch
import torch.nn as nn
import warnings
from typing import Optional
from transformers import BertModel


class Bert(nn.Module):
    def __init__(
            self,
            pretrained_path: str = "data/weights/bert-base-uncased",
            freeze_model: bool = True,
            hidden_state_index: int = -1,
            gradient_checkpointing: bool = False,
            attn_implementation: Optional[str] = None,
    ):
        super(Bert, self).__init__()

        self.hidden_state_index = hidden_state_index
        self.freeze_model = bool(freeze_model)

        from_pretrained_kwargs = {}
        if attn_implementation:
            from_pretrained_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = BertModel.from_pretrained(pretrained_path, **from_pretrained_kwargs)
        except TypeError:
            if "attn_implementation" in from_pretrained_kwargs:
                warnings.warn(
                    "BertModel.from_pretrained does not accept attn_implementation; fallback to default attention.",
                    RuntimeWarning,
                )
                from_pretrained_kwargs.pop("attn_implementation")
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
            with torch.no_grad():
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
