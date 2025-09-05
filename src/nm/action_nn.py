import math

import torch
from torch import nn


class FlatDANN(nn.Module):
    """
    A simple linear Action network which takes the input and memory, and produces the action/output.
    """

    def __init__(self, input_dim: int, memory_dim: int, target_dim: int, use_swish: bool = False):
        super().__init__()
        self.input_dim: int = input_dim
        self.memory_dim: int = memory_dim
        self.target_dim: int = target_dim

        flat_size = self.input_dim + self.memory_dim

        self.linear: nn.Linear = nn.Linear(flat_size, flat_size)
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(flat_size)
        if use_swish:
            self.relu: nn.SiLU = nn.SiLU()
        else:
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


class EmbDANN(nn.Module):
    """
    TODO
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


class ActionNN(nn.Module):
    """
    TODO
    """

    def __init__(self, memory_dim: int, target_dim: int, use_softmax: bool = False, input_dim: int = 0,
                 input_embedding_dim: int = 0, activation_fn_is_tanh: bool = False,
                 emb_activation_fn_is_tanh: bool = False):
        super().__init__()
        assert memory_dim > 0
        self.memory_dim: int = memory_dim
        assert target_dim > 0
        self.target_dim: int = target_dim
        self.use_softmax: bool = use_softmax
        assert input_dim >= 0
        self.input_dim: int = input_dim
        assert input_embedding_dim >= 0
        self.input_embedding_dim: int = input_embedding_dim
        if self.input_dim == 0:
            self.input_embedding_dim = 0

        everything_dim = self.memory_dim
        if self.input_embedding_dim > 0:
            self.emb_linear: nn.Linear = nn.Linear(self.input_dim, self.input_embedding_dim)
            self.emb_norm: nn.BatchNorm1d = nn.BatchNorm1d(self.input_embedding_dim)
            self.emb_activation_function: nn.Module = nn.Tanh() if emb_activation_fn_is_tanh else nn.ReLU()
            everything_dim += self.input_embedding_dim
        elif self.input_dim > 0:
            everything_dim += self.input_dim

        self.linear: nn.Linear = nn.Linear(everything_dim, target_dim)
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(target_dim)
        if self.use_softmax:
            self.softmax:  nn.Softmax = nn.Softmax(dim=1)
        else:
            self.activation_function: nn.Module = nn.Tanh() if activation_fn_is_tanh else nn.ReLU()

    def forward(self, input_symbol, memory):
        """
        Select an action/output.

        :param input_symbol: represents input at time t
        :param memory: represents memory at time t
        :return: output at time t + 1
        """
        if self.input_embedding_dim > 0:
            input_tensor = self.emb_linear(input_symbol)
            input_tensor = self.emb_norm(input_tensor)
            input_tensor = self.emb_activation_function(input_tensor)
            flat = torch.cat((input_tensor, memory), dim=1)
        elif self.input_dim > 0:
            flat = torch.cat((input_symbol, memory), dim=1)
        else:
            flat = memory

        flat = self.linear(flat)
        flat = self.norm(flat)
        if self.use_softmax:
            outputs = self.softmax(flat)
        else:
            outputs = self.activation_function(flat)
        return outputs


class ManyActionNN(nn.Module):
    """
    TODO comments
    """

    class LayerProperty:
        def __init__(self, in_dim: int = 0, out_dim: int = 0, num_chunks: int = 0,
                     use_embedding: bool = False, use_softmax: bool = False):
            self.in_dim: int = in_dim
            self.out_dim: int = out_dim
            self.num_chunks: int = num_chunks
            self.use_embedding: bool = use_embedding
            self.use_softmax: bool = use_softmax

        def chunk_in_dim(self):
            return self.in_dim // self.num_chunks

        def chunk_out_dim(self):
            return self.out_dim // self.num_chunks

        def __str__(self):
            return "LayerProperty(in_dim={}, out_dim={}, num_chunks={}, use_embedding={}, use_softmax={})".format(
                self.in_dim, self.out_dim, self.num_chunks, self.use_embedding, self.use_softmax
            )

        def __repr__(self):
            return self.__str__()

    def __init__(self, input_dim: int, input_embedding_dim: int, memory_dim: int, chunk_dim: int, target_dim: int,
                 activation_fn_is_tanh: bool = False, emb_activation_fn_is_tanh: bool = False):
        super().__init__()
        assert input_dim > 0 and input_embedding_dim > 0
        self.input_dim: int = input_dim
        self.input_embedding_dim: int = input_embedding_dim
        assert memory_dim > 0 and math.log2(memory_dim).is_integer()
        self.memory_dim: int = memory_dim
        assert chunk_dim > 0 and math.log2(chunk_dim).is_integer()
        self.chunk_dim: int = chunk_dim
        assert target_dim > 0
        self.target_dim: int = target_dim

        self.layer_properties: list[ManyActionNN.LayerProperty] = []
        layer_input_dim: int = self.memory_dim
        while layer_input_dim > self.chunk_dim:
            num_chunks = layer_input_dim // (2 * self.chunk_dim)
            layer_output_dim = num_chunks * self.chunk_dim
            self.layer_properties.append(
                ManyActionNN.LayerProperty(in_dim=layer_input_dim, out_dim=layer_output_dim, num_chunks=num_chunks)
            )
            layer_input_dim = layer_output_dim
        self.layer_properties.append(
            ManyActionNN.LayerProperty(in_dim=layer_input_dim, out_dim=target_dim, num_chunks=1)
        )
        self.layer_properties[0].use_embedding = True
        self.layer_properties[-1].use_softmax = True

        self.action_nets: nn.ModuleList = nn.ModuleList()
        for properties in self.layer_properties:
            layer_action_nets: nn.ModuleList = nn.ModuleList()
            for i in range(properties.num_chunks):
                layer_action_nets.append(
                    ActionNN(
                        properties.chunk_in_dim(), properties.chunk_out_dim(), use_softmax=properties.use_softmax,
                        input_dim=self.input_dim if properties.use_embedding else 0,
                        input_embedding_dim=self.input_embedding_dim if properties.use_embedding else 0,
                        activation_fn_is_tanh=activation_fn_is_tanh,
                        emb_activation_fn_is_tanh=emb_activation_fn_is_tanh
                    )
                )
            self.action_nets.append(layer_action_nets)

    def forward(self, input_symbol, memory):
        """
        Select an action/output.

        :param input_symbol: represents input at time t
        :param memory: represents memory at time t
        :return: output at time t + 1
        """
        intermediate = memory
        for i in range(len(self.layer_properties)):
            properties = self.layer_properties[i]
            chunks = torch.chunk(intermediate, properties.num_chunks, dim=1)
            next_chunks: list[torch.Tensor] = []
            layer_action_nets = self.action_nets[i]
            for j in range(len(layer_action_nets)):
                next_chunks.append(layer_action_nets[j](input_symbol, chunks[j]))
            if len(next_chunks) == 1:
                intermediate = next_chunks[0]
            else:
                intermediate = torch.cat(next_chunks, dim=1)
        return intermediate
