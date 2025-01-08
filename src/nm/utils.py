import numpy as np
import torch

from nm import state


def update_batch_history(config: dict, batch: tuple[list[int], np.ndarray, np.ndarray],
                         history: state.StateHistory) -> None:
    """
    Updates the current state with batch data.
    The memory and memory targets are not touched.
    """
    subset, inputs, targets = batch
    inputs: torch.Tensor = torch.from_numpy(inputs).to(config["device"])
    targets: torch.Tensor = torch.from_numpy(targets).to(config["device"])
    history.current().subset = subset
    history.current().update_inputs(inputs)
    history.current().update_targets(targets)
