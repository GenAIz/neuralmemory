import numpy as np


class Sequence:
    """
    Represents a sequence of inputs and targets, or input/output.
    The sequence is indexed with each index having an input and target.
    """

    def __init__(self):
        self.index: int = 0

    def restart(self) -> None:
        """
        Restart the sequence at the beginning
        """
        self.index = 0

    def next(self, include_source: bool = False) -> tuple:
        """
        Fetch the next input and target.  The input and target are transformed prior
        to return.  This allows input or target symbols to be transformed into vectors/embeddings.
        For example, a token could be transformed into its embedding prior to use.

        :param include_source: if true, the input and targets are tuples containing source symbols and transformations
        :return: input and target
        """
        if self.exhausted():
            raise ValueError("Sequence is exhausted; index out of bounds.")
        input_ = self.transform_inputs(self.index, include_source=include_source)
        target = self.transform_targets(self.index, include_source=include_source)
        self.index += 1
        return input_, target

    def transform_inputs(self, index: int, include_source: bool = False) -> np.ndarray:
        """
        Transform the inputs at index into a numpy.ndarray.
        If include_source is True then a tuple is returned with source symbol and its transformation
        """
        raise NotImplementedError()

    def transform_targets(self, index: int, include_source: bool = False) -> np.ndarray:
        """
        Transform the targets at index into a numpy.ndarray.
        If include_source is True then a tuple is returned with source symbol and its transformation
        """
        raise NotImplementedError()

    def output_to_symbol(self, index: int):
        """
        Transforms an output to a source symbol or raises a not implemented exception.
        :param index:  sequence index for which the source symbol is requested
        :return:  the source symbol at index
        """
        raise NotImplementedError()

    def exhausted(self) -> bool:
        """Returns a true if this sequence is exhausted"""
        raise NotImplementedError()

    def sizes(self) -> tuple[int, ...]:
        """Returns the sizes (dimensions) of the input and target (post transformation) as a tuple of ints."""
        raise NotImplementedError()

    def types(self) -> tuple[np.dtype, np.dtype]:
        """Returns the types of the input and target (post transformation) as a tuple of dtypes."""
        raise NotImplementedError()


class SequenceBatch:
    """
    Represents a batch of sequences.

    This differs a little from the conventional batch in that batches are not padded.
    Instead, when one sequence is exhausted another takes its place.  This results
    in a constant stream.

    If there are not enough sequences to make a full batch then the batch gets smaller.
    """

    def __init__(self, batch_size: int, allow_batch_size_of_one: bool = True):
        assert batch_size > 0, "Batch size must be greater than 0"
        self.batch_size: int = batch_size
        # can't have batch size of one for some NNs because it causes an error
        # example: batch norm expects more than 1 example to compute norm during training
        self.allow_batch_size_of_one = allow_batch_size_of_one
        self.sizes_: None | tuple[int, int] = None   # size of input and target
        self.dtypes: None | tuple[np.dtype, np.dtype] = None   # dtypes of input and target
        self.sequence_group_ids: list[int] = []
        self.sequences: list[Sequence] = []

    def output_to_symbol(self, index: int):
        """
        Converts an output index to a symbol, if possible.
        If not possible then raises a not implemented error.
        Calls output_to_symbol of the first known sequence.

        :param index:  sequence index for which the symbol is requested
        :return:  the symbol at index
        """
        return self.sequences[0].output_to_symbol(index)

    def next_random_sequence(self) -> Sequence | None:
        """Return a random sequence to be integrated into the batch, or None if none are available."""
        raise NotImplementedError()

    def restart(self, batch_size: int = -1, allow_batch_size_of_one: bool | None = None,
                is_training: bool = False) -> None:
        """
        Reset and prepare for first batch.

        :param batch_size:  the size of the batch else the default
        :param allow_batch_size_of_one:  if batch size of 1 is allowed
        :param is_training:  true if this is a training batch
        """
        if batch_size > 0:
            self.batch_size = batch_size
        if allow_batch_size_of_one is not None:
            self.allow_batch_size_of_one = allow_batch_size_of_one
        self.sequence_group_ids.clear()
        self.sequences.clear()
        while len(self.sequences) < self.batch_size:
            s = self.next_random_sequence()
            if s is None:
                break
            if self.sequence_group_ids:
                self.sequence_group_ids.append(self.sequence_group_ids[-1] + 1)
            else:
                self.sequence_group_ids.append(0)
            self.sequences.append(s)

    def exhausted(self) -> bool:
        """Return true if this sequence of batches is exhausted"""
        return not self.sequences or (len(self.sequences) == 1 and not self.allow_batch_size_of_one)

    def replace_exhausted_sequences(self) -> None:
        """Remove exhausted sequences from the batch and replace them with new ones."""
        has_none = False
        for i in range(len(self.sequences)):
            if self.sequences[i].exhausted():
                self.sequences[i] = self.next_random_sequence()
                has_none = has_none or self.sequences[i] is None
        if has_none:
            reduced_ids = []
            reduced = []
            for i in range(len(self.sequences)):
                if self.sequences[i] is not None:
                    reduced_ids.append(self.sequence_group_ids[i])
                    reduced.append(self.sequences[i])
            self.sequence_group_ids = reduced_ids
            self.sequences = reduced

    def next(self, include_source: bool = False) -> None | tuple[list[int], np.ndarray | tuple, np.ndarray | tuple]:
        """
        Returns a batch or None if all sequences have been exhausted.  A batch
        includes the sequence group IDs, inputs and targets.  Sequence group IDs
        is an ID for all sequences at position x within the batch, or None if the batch is complete.
        The ID helps relate sequences to each other when the batch size changes.  For example,
        if the batch size becomes smaller one can figure out which sequences were exhausted and
        how non-exhausted sequences relate to the previous batch of sequences.

        If include_source is true then returns inputs and targets as tuples where the first
        element of the tuple is source symbols
        """
        self.replace_exhausted_sequences()
        if not self.sequences:
            return None
        if not self.allow_batch_size_of_one and len(self.sequences) == 1:
            return None
        rows = len(self.sequences)
        input_sources = []
        target_sources = []
        inputs = np.zeros((rows, self.sizes_[0]), dtype=self.dtypes[0])
        targets = np.zeros((rows, self.sizes_[1]), dtype=self.dtypes[1])
        for row in range(rows):
            if include_source:
                inputs_with_source, targets_with_source = self.sequences[row].next(include_source=True)
                inputs[row, :], targets[row, :] = inputs_with_source[1], targets_with_source[1]
                input_sources.append(inputs_with_source[0])
                target_sources.append(targets_with_source[0])
            else:
                inputs[row, :], targets[row, :] = self.sequences[row].next(include_source=False)
        group_ids = None if self.batch_size == len(self.sequence_group_ids) else self.sequence_group_ids
        if include_source:
            return group_ids, (input_sources, inputs), (target_sources, targets)
        return group_ids, inputs, targets

    def sizes(self) -> tuple[int, int] | None:
        """
        Returns the sizes of the sequence as a tuple of ints.
        Valid only after restart, otherwise can return None.
        """
        return self.sizes_

    def types(self) -> tuple[np.dtype, np.dtype]:
        """
        Returns the types of the input and target as a tuple of dtypes.
        Valid only after restart, otherwise can return None.
        """
        return self.dtypes
