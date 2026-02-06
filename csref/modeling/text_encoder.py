import torch
import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):
    def __init__(
            self,
            pretrained_path: str = "data/weights/bert-base-uncased",
            freeze_model: bool = True,
            hidden_state_index: int = -1,
            gradient_checkpointing: bool = False,
    ):
        super(Bert, self).__init__()

        self.hidden_state_index = hidden_state_index

        self.model = BertModel.from_pretrained(pretrained_path)

        if freeze_model:
            self.frozen(self.model)

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

    def forward(self, text_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        need_hidden_states = self.hidden_state_index != -1
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

        return feat
