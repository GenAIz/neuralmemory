import torch
from torch import nn


class MemoryNN(nn.Module):
    """
    A simple linear memory-net which takes the input, action and memory, and produces the next memory.
    """

    def __init__(self, input_dim: int, target_dim: int, memory_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.target_dim: int = target_dim
        self.memory_dim: int = memory_dim

        flat_size = self.input_dim + self.target_dim + self.memory_dim
        self.linear: nn.Linear = nn.Linear(flat_size, self.memory_dim)
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(self.memory_dim)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input_symbol, action, memory):
        """
        Update the memory given the latest input and action/output.

        :param input_symbol: represents the input at time t
        :param action: represents the output at time t
        :param memory: represents the memory at time t
        :return: the memory at time t + 1
        """
        flat = torch.cat((input_symbol, action, memory), dim=1)
        flat = self.linear(flat)
        flat = self.norm(flat)
        flat = self.sigmoid(flat)

        return flat


class Memory2NN(nn.Module):
    """
    A simple linear memory-net which takes the input, action and memory, and produces the next memory.
    This net has an extra layer compared to MemoryNN.
    """

    def __init__(self, input_dim: int, target_dim: int, memory_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.target_dim: int = target_dim
        self.memory_dim: int = memory_dim

        flat_size = self.input_dim + self.target_dim + self.memory_dim

        self.linear1: nn.Linear = nn.Linear(flat_size, self.memory_dim)
        self.norm1: nn.BatchNorm1d = nn.BatchNorm1d(self.memory_dim)
        self.sigmoid1: nn.Sigmoid = nn.Sigmoid()

        self.linear2: nn.Linear = nn.Linear(flat_size, self.memory_dim)
        self.norm2: nn.BatchNorm1d = nn.BatchNorm1d(self.memory_dim)
        self.sigmoid2: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input_symbol, action, memory):
        """
        Update the memory given the latest input and action/output.

        :param input_symbol: represents the input at time t
        :param action: represents the output at time t
        :param memory: represents the memory at time t
        :return: the memory at time t + 1
        """
        flat = torch.cat((input_symbol, action, memory), dim=1)
        flat = self.linear1(flat)
        flat = self.norm1(flat)
        flat = self.sigmoid1(flat)

        flat = torch.cat((input_symbol, action, flat), dim=1)
        flat = self.linear2(flat)
        flat = self.norm2(flat)
        flat = self.sigmoid2(flat)

        return flat


class Memory3NN(nn.Module):
    """
    A simple linear memory-net which takes the input, action and memory, and produces the next memory.
    This net has two extra layers compared to MemoryNN.
    """

    def __init__(self, input_dim: int, target_dim: int, memory_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.target_dim: int = target_dim
        self.memory_dim: int = memory_dim

        flat_size = self.input_dim + self.target_dim + self.memory_dim

        self.linear1: nn.Linear = nn.Linear(flat_size, self.memory_dim)
        self.norm1: nn.BatchNorm1d = nn.BatchNorm1d(self.memory_dim)
        self.sigmoid1: nn.Sigmoid = nn.Sigmoid()

        self.linear2: nn.Linear = nn.Linear(flat_size, self.memory_dim)
        self.norm2: nn.BatchNorm1d = nn.BatchNorm1d(self.memory_dim)
        self.sigmoid2: nn.Sigmoid = nn.Sigmoid()

        self.linear3: nn.Linear = nn.Linear(flat_size, self.memory_dim)
        self.norm3: nn.BatchNorm1d = nn.BatchNorm1d(self.memory_dim)
        self.sigmoid3: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input_symbol, action, memory):
        """
        Update the memory given the latest input and action/output.

        :param input_symbol: represents the input at time t
        :param action: represents the output at time t
        :param memory: represents the memory at time t
        :return: the memory at time t + 1
        """
        flat = torch.cat((input_symbol, action, memory), dim=1)
        flat = self.linear1(flat)
        flat = self.norm1(flat)
        flat = self.sigmoid1(flat)

        flat = torch.cat((input_symbol, action, flat), dim=1)
        flat = self.linear2(flat)
        flat = self.norm2(flat)
        flat = self.sigmoid2(flat)

        flat = torch.cat((input_symbol, action, flat), dim=1)
        flat = self.linear3(flat)
        flat = self.norm3(flat)
        flat = self.sigmoid3(flat)

        return flat


class MemEmbMNN(nn.Module):
    """
    Similar to Memory2NN with the input and action converted to an embedding before use.
    """

    def __init__(self, input_dim: int, target_dim: int, memory_dim: int, embedding_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.target_dim: int = target_dim
        self.memory_dim: int = memory_dim
        self.embedding_dim: int = embedding_dim

        assert self.input_dim == self.target_dim
        self.linear1: nn.Linear = nn.Linear(self.input_dim, self.embedding_dim)
        self.norm1: nn.BatchNorm1d = nn.BatchNorm1d(self.embedding_dim)
        self.relu1: nn.ReLU = nn.ReLU()

        flat_size = self.embedding_dim + self.embedding_dim + self.memory_dim
        self.linear2: nn.Linear = nn.Linear(flat_size, self.memory_dim)
        self.norm2: nn.BatchNorm1d = nn.BatchNorm1d(self.memory_dim)
        self.sigmoid2: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input_symbol, action, memory):
        """
        Update the memory given the latest input and action/output.

        :param input_symbol: represents the input at time t
        :param action: represents the output at time t
        :param memory: represents the memory at time t
        :return: the memory at time t + 1
        """
        input_embedding = self.linear1(input_symbol)
        input_embedding = self.norm1(input_embedding)
        input_embedding = self.relu1(input_embedding)

        action_embedding = self.linear1(action)
        action_embedding = self.norm1(action_embedding)
        action_embedding = self.relu1(action_embedding)

        flat = torch.cat((input_embedding, action_embedding, memory), dim=1)
        flat = self.linear2(flat)
        flat = self.norm2(flat)
        flat = self.sigmoid2(flat)

        return flat

