import torch
import torch.nn as nn
from typing import Optional, Dict

from transformers import Wav2Vec2Model
from transformers.activations import GELUActivation

from .layers import AttFlat


class GELUConv(nn.Module):
    """A Conv2d -> Batchnorm -> GELU activation"""

    def __init__(
            self, in_channels: int, out_channels: int, ksize: int, stride: int, 
            shortcut: bool = False, groups: int = 1, bias: bool = False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = GELUActivation()

        self.add = shortcut and in_channels == out_channels
        if shortcut:
            assert in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.add:
            out = x + out
        return self.act(self.bn(out))


def make_mask(feature: torch.Tensor) -> torch.Tensor:
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class Wav2vec2(nn.Module):
    def __init__(
            self,
            hidden_size: int = 1024,
            flat_glimpses: int = 1,
            dropout_rate: float = 0.1,
            target_sr: int = 16000,
            pretrained_path: str = "data/weights/wav2vec2-base-960h",
            freeze_model: bool = False,
            use_one_hidden_state_as_feat: bool = True,
            hidden_state_index: int = -1,
            use_att_flat_mask: bool = True,
            fusion_times: int = 1,
            freeze_layers: Optional[int] = None,  # freeze the first few layers
            short_cut: bool = False,
            gradient_checkpointing: bool = False,
    ):
        super(Wav2vec2, self).__init__()

        self.hidden_size = hidden_size
        self.target_sample_rate = target_sr
        self.use_one_hidden_state_as_feat = use_one_hidden_state_as_feat
        self.hidden_state_index = hidden_state_index
        self.use_att_flat_mask = use_att_flat_mask
        self.fusion_times = fusion_times

        self.model = Wav2Vec2Model.from_pretrained(pretrained_path, gradient_checkpointing=gradient_checkpointing)

        if freeze_model:
            if isinstance(freeze_layers, bool):
                freeze_layers = len(self.model.encoder.layers) if freeze_layers else None

            if freeze_layers is None:
                # Legacy behavior: only freeze the feature encoder.
                self.model.freeze_feature_encoder()
            else:
                if not isinstance(freeze_layers, int):
                    raise TypeError("freeze_layers must be int or null")
                if freeze_layers < 0:
                    raise ValueError("freeze_layers must be >= 0")

                if hasattr(self.model, "masked_spec_embed"):
                    self.model.masked_spec_embed.requires_grad = False
                self.frozen(self.model.feature_extractor)
                self.frozen(self.model.feature_projection)
                self.frozen(self.model.encoder.pos_conv_embed)
                self.frozen(self.model.encoder.layer_norm)
                for layer in self.model.encoder.layers[:freeze_layers]:
                    self.frozen(layer)

        if not use_one_hidden_state_as_feat:
            self.fusion_modules = GELUConv(
                in_channels=abs(hidden_state_index),
                out_channels=fusion_times,
                ksize=1,
                stride=1,
                shortcut=short_cut
            )
        self.att_flat = AttFlat(hidden_size, flat_glimpses, dropout_rate)

    def frozen(self, module: nn.Module):
        if hasattr(module, 'module'):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, audio: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:

        output = self.model(
            input_values=audio,
            attention_mask=mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        feat = None
        if self.use_one_hidden_state_as_feat:
            hidden_state = output.hidden_states[self.hidden_state_index]  # [b, len, c]
            feat = hidden_state
        else:
            feat = torch.stack(output.hidden_states[self.hidden_state_index:], 1)  # [b, n_hidden, len, c]
            feat = self.fusion_modules(feat)  # [b, 1, len, c]
            feat = torch.flatten(feat, 1, 2)  # [b, len, c]

        mask_flip_bool = None
        if self.use_att_flat_mask:
            # Note: access private method _get_feature_vector_attention_mask. 
            # In transformers > 4.26, this logic might change. Keeping it for now but worth monitoring.
            last_hidden_len = output.hidden_states[-1].shape[1]
            first_attention_0 = self.model._get_feature_vector_attention_mask(last_hidden_len, mask)  # [b, len]
            mask_flip_bool = make_mask(first_attention_0.unsqueeze(2))  # [b, 1, 1, len]
            
            if self.fusion_times > 1:
                # Original: torch.cat([mask_flip_bool for i in range(self.fusion_times)], 3)
                mask_flip_bool = mask_flip_bool.repeat(1, 1, 1, self.fusion_times)

        flat_feat = self.att_flat(feat, mask_flip_bool)

        return {
            'flat_feat': flat_feat,
            'feat': feat,
        }
