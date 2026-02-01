import torch
import torch.nn as nn
import numpy as np


# ContrastiveSemanticAlignment (CSA) pretraing stage for speech encoder
class ContrastiveSemanticAlignment(nn.Module):
    def __init__(
            self,
            speech_encoder: nn.Module,
            text_encoder: nn.Module,
    ):
        super(ContrastiveSemanticAlignment, self).__init__()
        self.speech_encoder = speech_encoder
        self.text_encoder = text_encoder

        self.contrastive_loss = CLIPLoss1D()

    def frozen(self, module):
        if hasattr(module, 'module'):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, audios, audios_mask, text_ids, text_ids_mask):

        # speech and text encoding
        x = self.speech_encoder(audios, audios_mask)
        y = self.text_encoder(text_ids, text_ids_mask)

        x = x['flat_feat']

        loss = self.contrastive_loss(x, y)

        return loss


# reference: https://github.com/descriptinc/lyrebird-wav2clip
class CLIPLoss1D(nn.Module):
    def __init__(self):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_audio = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()

    def forward(self, audio_features, text_features):
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * audio_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ audio_features.t()

        batch_size = audio_features.shape[0]
        # Ensure ground_truth is on the same device as features
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=audio_features.device)
        
        return (
                       self.loss_audio(logits_per_image, ground_truth)
                       + self.loss_text(logits_per_text, ground_truth)
               ) / 2
