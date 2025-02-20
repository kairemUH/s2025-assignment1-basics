import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    """
    Implementation of the AdamW optimizer.

    Args:
        params (Iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float): Weight decay coefficient (L2 regularization) (default: 0.01).

    Reference:
        - "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2018)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (Optional[Callable]): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # Initialize state variables if not present
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Update step
                state["step"] += 1
                step = state["step"]

                # Compute bias-corrected learning rates
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Update first moment estimate (mean of gradients)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update second moment estimate (uncentered variance of gradients)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected moment estimates
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                # Apply weight decay separately from Adam update
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    Computes the learning rate using a cosine schedule with a warmup phase.

    Args:
        t (int): Current iteration step.
        alpha_max (float): Maximum learning rate.
        alpha_min (float): Minimum learning rate.
        T_w (int): Number of warmup iterations.
        T_c (int): Number of cosine annealing iterations.

    Returns:
        float: The computed learning rate for step t.
    """
    if t < T_w:
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min

def clip_gradients(params, max_norm, epsilon=1e-6):
    """
    Clips the gradients of a list of parameters to a maximum L2-norm.

    Args:
        params (iterable): Iterable of torch.nn.Parameter with gradients.
        max_norm (float): Maximum allowed L2 norm for gradients.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        None (modifies gradients in place).
    """
    total_norm = torch.sqrt(sum(p.grad.norm(2).pow(2) for p in params if p.grad is not None) + epsilon)

    # Compute scaling factor
    clip_coef = max_norm / (total_norm + epsilon)
    clip_coef = min(1.0, clip_coef)

    # Scale gradients if necessary
    for p in params:
        if p.grad is not None:
            p.grad.mul_(clip_coef)  # In-place modification