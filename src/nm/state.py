import numpy as np
import torch


def _zeros(dim: int | tuple[int, ...], dtype: np.dtype | torch.dtype,
           device: str | None = None) -> np.ndarray | torch.Tensor:
    """
    Creates the appropriate tensor based on the dtype.
    :param dim:  tensor size
    :param dtype:   tensor type
    :param device:  device, if any
    :return: a tensor of zeroes
    """
    if isinstance(dtype, torch.dtype):
        return torch.zeros(dim, dtype=dtype, device=device)
    else:
        return np.zeros(dim, dtype=dtype)


def _zeros_like(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Creates the appropriate tensor based on the given tensor.
    :param x:  tensor to copy
    :return: a tensor of zeroes
    """
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    else:
        return np.zeros_like(x)


def _copy(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Copy a tensor.
    :param x:  tensor to copy
    :return: a copy of the tensor
    """
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    else:
        return np.copy(x)


class State:
    """
    Represents a state within the computational sequence at time t.
    This is equivalent to an RL state for experience replay.

    The state captures input, action, target, memory and the loss at time t.  The action
    is the output of the action network and the target is the expected output.  The target
    is included for completeness but isn't typically used.

    Note that the memory is the output of the memory-net at time t - 1
    because the t - 1 memory is used to predict the memory at time t.
    """

    def __init__(self, input_dim: int, input_dtype: np.dtype | torch.dtype,
                 memory_dim: int, memory_dtype: np.dtype | torch.dtype,
                 target_dim: int, target_dtype: np.dtype | torch.dtype,
                 device: str | None = None):
        self.input: np.ndarray | torch.Tensor = _zeros(input_dim, input_dtype, device=device)
        self.memory: np.ndarray | torch.Tensor = _zeros(memory_dim, memory_dtype, device=device)
        self.action: np.ndarray | torch.Tensor = _zeros(target_dim, target_dtype, device=device)
        self.target: np.ndarray | torch.Tensor = _zeros(target_dim, target_dtype, device=device)
        self.action_loss: float | np.float32 | torch.float32 = 0
        if isinstance(target_dtype, torch.dtype):
            self.action_loss = torch.tensor(0, dtype=target_dtype, device=device)

    def reset(self) -> None:
        """Reset the state to 0s"""
        self.input = _zeros_like(self.input)
        self.memory = _zeros_like(self.memory)
        self.action = _zeros_like(self.target)
        self.target = _zeros_like(self.target)
        if isinstance(self.target, torch.Tensor):
            self.action_loss = torch.tensor(0, dtype=self.target.dtype, device=self.target.device)
        else:
            self.action_loss = 0


EMPTY_SIZE_OR_SHAPE = tuple()


def _dim0(x: list | np.ndarray | torch.Tensor) -> int:
    """Return the size of the first or 0 dimension"""
    if isinstance(x, list):
        return len(x)
    elif isinstance(x, torch.Tensor):
        if x.size() == EMPTY_SIZE_OR_SHAPE:
            return 0
        else:
            return x.size(0)
    else:
        if x.shape == EMPTY_SIZE_OR_SHAPE:
            return 0
        else:
            return x.shape[0]


class BatchState:
    """
    A batch of states.

    At any particular point in time a sequence's state may be captured.  When training
    with a batch of sequences, a batch of corresponding states is captured.
    """

    def __init__(self, size: int,
                 input_dim: int, input_dtype: np.dtype | torch.dtype,
                 memory_dim: int, memory_dtype: np.dtype | torch.dtype,
                 target_dim: int, target_dtype: np.dtype | torch.dtype,
                 subset: list[int] | np.ndarray | torch.Tensor | None,
                 device: str | None = None):
        assert size > 0
        self.states = []
        for i in range(size):
            self.states.append(
                State(input_dim, input_dtype, memory_dim, memory_dtype, target_dim, target_dtype, device=device)
            )
        self.subset: list[int] | np.ndarray | torch.Tensor | None = subset

    def _update(self, update_fn, x: np.ndarray | torch.Tensor,
                override: list[int] | np.ndarray | torch.Tensor | None = None) -> None:
        """
        Update the state by applying the update function update_fn to the
        tensors.  Updating o subset of states is possible by providing the
        override indexes - only the ones to be treated.

        :param update_fn:  update function
        :param x:  the new value(s)
        :param override:  the subset to update, or None which implies update all
        """
        subset = self.subset if override is None else override
        size = len(self.states) if subset is None else _dim0(subset)
        for i in range(size):
            if self.subset is None:
                update_fn(self.states[i], x[i, :])
            else:
                update_fn(self.states[subset[i]], x[i, :])

    def update_inputs(self, inputs: np.ndarray | torch.Tensor) -> None:
        def update(state: State, input: np.ndarray | torch.Tensor):
            state.input = input
        self._update(update, inputs)

    def update_memories(self, memories: np.ndarray | torch.Tensor) -> None:
        def update(state: State, memory: np.ndarray | torch.Tensor):
            state.memory = memory
        self._update(update, memories)

    def update_actions(self, actions: np.ndarray | torch.Tensor) -> None:
        def update(state: State, action: np.ndarray | torch.Tensor):
            state.action = action
        self._update(update, actions)

    def update_targets(self, targets: np.ndarray | torch.Tensor) -> None:
        def update(state: State, target: np.ndarray | torch.Tensor):
            state.target = target
        self._update(update, targets)

    def update_action_losses(self, action_losses: np.ndarray | torch.Tensor) -> None:
        def update(state: State, action_loss: np.ndarray | torch.Tensor):
            if isinstance(action_loss, torch.Tensor):
                if action_loss.size() == (1,):
                    state.action_loss = action_loss[0]
                elif action_loss.size() == (1, 1):
                    state.action_loss = action_loss[0, 0]
                else:
                    raise ValueError("Unexpected action loss size {}".format(action_loss.size()))
            else:
                if action_loss.shape == (1,):
                    state.action_loss = action_loss[0]
                elif action_loss.shape == (1, 1):
                    state.action_loss = action_loss[0, 0]
                else:
                    raise ValueError("Unexpected action loss shape {}".format(action_loss.shape))
        self._update(update, action_losses)

    def _reconstruct(self, get_fn,
                     override: list[int] | np.ndarray | torch.Tensor | None = None) -> np.ndarray | torch.Tensor:
        """
        Reconstruct a batch tensor using the get_fn function to retrieve the tensors to "glue" together.

        :param get_fn:  the function that retrieves the "sub" tensors
        :param override:  the subset to fetch, or None which implies fetch all
        :return:  a tensor that contains all "sub" tensors
        """
        subset = self.subset if override is None else override
        size = len(self.states) if subset is None else _dim0(subset)
        example = get_fn(self.states[0])
        device = example.device if isinstance(example, torch.Tensor) else None
        x = _zeros((size, max(1, _dim0(example))), dtype=example.dtype, device=device)
        for i in range(size):
            if self.subset is None:
                x[i, :] = get_fn(self.states[i])
            else:
                x[i, :] = get_fn(self.states[subset[i]])
        return x

    def inputs(self) -> np.ndarray | torch.Tensor:
        def get(state: State):
            return state.input
        return self._reconstruct(get)

    def memories(self) -> np.ndarray | torch.Tensor:
        def get(state: State):
            return state.memory
        return self._reconstruct(get)

    def actions(self) -> np.ndarray | torch.Tensor:
        def get(state: State):
            return state.action
        return self._reconstruct(get)

    def targets(self) -> np.ndarray | torch.Tensor:
        def get(state: State):
            return state.target
        return self._reconstruct(get)

    def action_losses(self, override: list[int] | np.ndarray | torch.Tensor | None = None) -> np.ndarray | torch.Tensor:
        def get(state: State):
            return state.action_loss
        return self._reconstruct(get, override=override)


class StateHistory:
    """
    A sequence or history of State or BatchState
    """

    def __init__(self, size: int, store_batches: bool = True):
        self.size = size
        self.store_batches = store_batches
        self.history: list[State | BatchState | None] = [None] * self.size

    def reset(self) -> None:
        """Reset the history to hold None"""
        self.history = [None] * self.size

    def has(self, past: int) -> bool:
        """Return true if the past item can be retrieved (history is long enough)"""
        return 1 + past <= self.size

    def add(self, state: State | BatchState) -> None:
        """Add a State or BatchState to the history"""
        assert isinstance(state, State) or isinstance(state, BatchState)
        assert (self.store_batches and isinstance(state, BatchState)) \
               or (not self.store_batches and isinstance(state, State))
        self.history.append(state)
        self.history.pop(0)

    def current(self) -> State | BatchState | None:
        """Return the current/last state or batch state"""
        return self.history[-1]

    def past(self, past: int) -> State | BatchState | None:
        """Return a past state or batch state"""
        return self.history[-1 - past]

    def ancient(self) -> State | BatchState | None:
        """Return the oldest state or batch state"""
        return self.history[0]
