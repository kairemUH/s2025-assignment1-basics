import torch
import numpy as np
import os
import typing

def load_data(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Generates input-target pairs for training a language model.

    Args:
        x (np.ndarray): 1D array of token IDs.
        batch_size (int): Number of sequences per batch.
        context_length (int): Length of each sequence.
        device (str): Device to place the tensors on ('cpu' or 'cuda:x').

    Returns:
        input_tensor (torch.Tensor): Tensor of shape (batch_size, context_length).
        target_tensor (torch.Tensor): Tensor of shape (batch_size, context_length).
    """
    # Randomly sample starting indices for sequences
    max_index = len(x) - context_length
    assert max_index > 0, "Dataset too small for the given context length."

    start_indices = np.random.randint(0, max_index, size=batch_size)

    # Construct input and target sequences
    input_batch = np.array([x[i : i + context_length] for i in start_indices])
    target_batch = np.array([x[i + 1 : i + context_length + 1] for i in start_indices])

    # Convert to PyTorch tensors and move to device
    input_tensor = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_batch, dtype=torch.long, device=device)

    return input_tensor, target_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: typing.Union[str, os.PathLike, typing.BinaryIO]):
    """
    Saves a checkpoint of the model and optimizer.

    Args:
        model (torch.nn.Module): The model whose state needs to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state needs to be saved.
        iteration (int): The current iteration number.
        out (str | os.PathLike | typing.BinaryIO): Path or file object to save the checkpoint.

    Returns:
        None
    """
    checkpoint = {
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)
    print(f"Checkpoint saved at {out}")


def load_checkpoint(src: typing.Union[str, os.PathLike, typing.BinaryIO], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Loads a checkpoint and restores model and optimizer states.

    Args:
        src (str | os.PathLike | typing.BinaryIO): Path or file object of the saved checkpoint.
        model (torch.nn.Module): The model where the state will be restored.
        optimizer (torch.optim.Optimizer): The optimizer where the state will be restored.

    Returns:
        int: The iteration number saved in the checkpoint.
    """
    checkpoint = torch.load(src, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]

    print(f"Checkpoint loaded from {src}, resuming at iteration {iteration}")
    return iteration