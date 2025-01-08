import torch
from torch import nn


class FlatDANN(nn.Module):
    """
    A simple linear Action network which takes the input and memory, and produces the action/output.
    """

    def __init__(self, input_dim: int, memory_dim: int, target_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.memory_dim: int = memory_dim
        self.target_dim: int = target_dim

        flat_size = self.input_dim + self.memory_dim

        self.linear: nn.Linear = nn.Linear(flat_size, flat_size)
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(flat_size)
        self.relu: nn.ReLU = nn.ReLU()

        self.decision: nn.Linear = nn.Linear(flat_size, target_dim)
        self.softmax:  nn.Softmax = nn.Softmax(dim=1)

    def forward(self, input_symbol, memory):
        """
        Select an action/output.

        :param input_symbol: represents input at time t
        :param memory: represents memory at time t
        :return: output at time t + 1
        """
        flat = torch.cat((input_symbol, memory), dim=1)
        flat = self.linear(flat)
        flat = self.norm(flat)
        flat = self.relu(flat)
        decided = self.decision(flat)
        return self.softmax(decided)


class Flat2DANN(nn.Module):
    """
    A simple linear Action network which takes the input and memory, and produces the action/output.
    This net has an extra layer compared to FlatDANN.
    """

    def __init__(self, input_dim: int, memory_dim: int, target_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.memory_dim: int = memory_dim
        self.target_dim: int = target_dim

        flat_size = self.input_dim + self.memory_dim

        self.linear1: nn.Linear = nn.Linear(flat_size, flat_size)
        self.norm1: nn.BatchNorm1d = nn.BatchNorm1d(flat_size)
        self.relu1: nn.ReLU = nn.ReLU()

        self.linear2: nn.Linear = nn.Linear(flat_size, flat_size)
        self.norm2: nn.BatchNorm1d = nn.BatchNorm1d(flat_size)
        self.relu2: nn.ReLU = nn.ReLU()

        self.decision: nn.Linear = nn.Linear(flat_size, target_dim)
        self.softmax:  nn.Softmax = nn.Softmax(dim=1)

    def forward(self, input_symbol, memory):
        """
        Select an action/output.

        :param input_symbol: represents input at time t
        :param memory: represents memory at time t
        :return: output at time t + 1
        """
        flat = torch.cat((input_symbol, memory), dim=1)

        flat = self.linear1(flat)
        flat = self.norm1(flat)
        flat = self.relu1(flat)

        flat = self.linear2(flat)
        flat = self.norm2(flat)
        flat = self.relu2(flat)

        decided = self.decision(flat)
        return self.softmax(decided)


class EmbDANN(nn.Module):
    """
    Similar to Flat2DANN with the input converted to an embedding before use.
    """

    def __init__(self, input_dim: int, memory_dim: int, target_dim: int, embedding_dim: int):
        super().__init__()
        self.input_dim: int = input_dim
        self.memory_dim: int = memory_dim
        self.target_dim: int = target_dim
        self.embedding_dim: int = embedding_dim

        self.emb_linear: nn.Linear = nn.Linear(self.input_dim, self.embedding_dim)
        self.emb_norm: nn.BatchNorm1d = nn.BatchNorm1d(self.embedding_dim)
        self.emb_relu: nn.ReLU = nn.ReLU()

        flat_size = self.embedding_dim + self.memory_dim

        self.linear1: nn.Linear = nn.Linear(flat_size, flat_size)
        self.norm1: nn.BatchNorm1d = nn.BatchNorm1d(flat_size)
        self.relu1: nn.ReLU = nn.ReLU()

        self.linear2: nn.Linear = nn.Linear(flat_size, flat_size)
        self.norm2: nn.BatchNorm1d = nn.BatchNorm1d(flat_size)
        self.relu2: nn.ReLU = nn.ReLU()

        self.decision: nn.Linear = nn.Linear(flat_size, target_dim)
        self.softmax:  nn.Softmax = nn.Softmax(dim=1)

    def forward(self, input_symbol, memory):
        """
        Select an action/output.

        :param input_symbol: represents input at time t
        :param memory: represents memory at time t
        :return: output at time t + 1
        """

        input_embedding = self.emb_linear(input_symbol)
        input_embedding = self.emb_norm(input_embedding)
        input_embedding = self.emb_relu(input_embedding)

        flat = torch.cat((input_embedding, memory), dim=1)

        flat = self.linear1(flat)
        flat = self.norm1(flat)
        flat = self.relu1(flat)

        flat = self.linear2(flat)
        flat = self.norm2(flat)
        flat = self.relu2(flat)

        decided = self.decision(flat)
        return self.softmax(decided)
