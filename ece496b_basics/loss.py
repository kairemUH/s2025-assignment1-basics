import torch
import torch.nn.functional as F

import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss for a batch of logits and corresponding targets.

    Args:
        logits (torch.Tensor): The predicted logits of shape (batch_size, vocab_size).
        targets (torch.Tensor): The ground truth indices of shape (batch_size).

    Returns:
        torch.Tensor: The averaged cross-entropy loss.
    """
    # Improve numerical stability by shifting logits to prevent large exponentials
    adjusted_logits = logits - logits.amax(dim=-1, keepdim=True)

    # Compute the negative log-probability of the target tokens
    target_log_prob = -adjusted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()

    # Compute the log sum of exponentials for normalization (log softmax trick)
    log_partition = torch.logsumexp(adjusted_logits, dim=-1)

    # Compute final loss and return mean over batch
    loss = target_log_prob + log_partition
    return loss.mean()