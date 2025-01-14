import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: torch.Tensor = None,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature, normalize):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, prediction_embs, target_embs, choices_dim=1, output_head=None, target_head=None):
        assert len(prediction_embs.shape) == len(target_embs.shape)
        assert all([s1==s2 for s1, s2 in zip(prediction_embs.shape, target_embs.shape)])
        if output_head is not None and target_head is None:
            all_embs = torch.cat((prediction_embs, target_embs), 0)
            all_embs = output_head(all_embs)
            prediction_embs, target_embs = torch.chunk(all_embs, chunks=2, dim=0)
        elif output_head is not None and target_head is not None:
            prediction_embs = output_head(prediction_embs)
            target_embs = target_head(target_embs)

        if self.normalize:
            prediction_embs = F.normalize(prediction_embs, p=2, dim=-1)
            target_embs = F.normalize(target_embs, p=2, dim=-1)

        if choices_dim < 0 :
            choices_dim += len(target_embs.shape)
        if choices_dim != len(target_embs.shape)-2: ## 1,B*128
            target_embs = target_embs.transpose(choices_dim, len(target_embs.shape)-2) ## transpose(1,0)  C*B
            prediction_embs = prediction_embs.transpose(choices_dim, len(target_embs.shape)-2)
        if len(target_embs.shape) != 3:
            sz = prediction_embs.shape ## B C
            prediction_embs = prediction_embs.view(-1, sz[-2], sz[-1]) ## 1 B C
            target_embs = target_embs.view(-1, sz[-2], sz[-1]) ## 1 B C

        # Compute scores
        scores = torch.bmm(prediction_embs, target_embs.transpose(-2, -1)) / self.temperature ## 1 B C 和 1 C B  B T C和B C T
        scores = scores.flatten(0, 1) ## B B ；  B T T——》B*T T

        # Labels
        bs, n_choices = target_embs.shape[:2]  ## 1 B ；  B T
        labels = torch.arange(n_choices, device=prediction_embs.device) # B
        if bs > 1:
            labels = torch.stack([labels] * bs, 0).flatten(0, 1)

        # Compute loss
        loss = F.cross_entropy(scores, labels)
        ## 这里B*T T和B*T做交叉熵。  或B B和B做交叉熵
        return loss, scores

