import sys

import torch
from torch import nn


class MemoryNN(nn.Module):
    """
    A simple linear memory-net which takes the input, action and memory, and produces the next memory.
    """

    def __init__(self, input_dim: int, memory_dim: int, input_embedding_dim: int = 0,
                 intermediate_layers: int = 0, intermediate_dim: int = 0, target_dim: int = 0,
                 memory_output_dim: int = 0, activation_fn_is_tanh: bool = False,
                 emb_activation_fn_is_tanh: bool = False):
        """
        TODO comments

        :param input_dim:
        :param memory_dim:
        :param input_embedding_dim:
        :param intermediate_layers:
        :param intermediate_dim:
        :param target_dim:
        :param memory_output_dim:
        """
        super().__init__()
        if input_dim <= 0 or memory_dim <= 0:
            raise ValueError("Input and memory dimensions cannot be 0 or less")
        if target_dim < 0:
            raise ValueError("Target dimension cannot be less than 0")
        if intermediate_layers != 0 and (intermediate_layers <= 0 or intermediate_dim <= 0):
            raise ValueError("Bad intermediate values.")
        self.input_dim: int = input_dim
        self.target_dim: int = target_dim
        self.memory_dim: int = memory_dim
        self.intermediate_layers: int = intermediate_layers
        self.intermediate_dim: int = intermediate_dim
        if self.intermediate_layers == 0:
            self.intermediate_dim = 0
        self.input_embedding_dim = input_embedding_dim
        self.memory_output_dim: int = memory_output_dim

        if self.input_embedding_dim > 0:
            self.emb_linear: nn.Linear = nn.Linear(self.input_dim, self.input_embedding_dim)
            self.emb_norm: nn.BatchNorm1d = nn.BatchNorm1d(self.input_embedding_dim)
            self.emb_activation_function: nn.Module = nn.Tanh() if emb_activation_fn_is_tanh else nn.ReLU()

        everything_dim = self.target_dim + self.memory_dim + self.intermediate_dim
        if self.input_embedding_dim > 0:
            everything_dim += self.input_embedding_dim
        else:
            everything_dim += self.input_dim

        if self.intermediate_layers > 0:
            self.layers: nn.ModuleList = nn.ModuleList()
            first_inter_dim = self.target_dim + self.memory_dim
            if self.input_embedding_dim > 0:
                first_inter_dim += self.input_embedding_dim
            else:
                first_inter_dim += self.input_dim
            self.layers.append(nn.Linear(first_inter_dim, self.intermediate_dim))
            self.layers.append(nn.BatchNorm1d(self.intermediate_dim))
            self.layers.append(nn.Tanh() if activation_fn_is_tanh else nn.ReLU())
            for i in range(1, self.intermediate_layers):
                self.layers.append(nn.Linear(everything_dim, self.intermediate_dim))
                self.layers.append(nn.BatchNorm1d(self.intermediate_dim))
                self.layers.append(nn.Tanh() if activation_fn_is_tanh else nn.ReLU())

        mem_out_dim = self.memory_dim if self.memory_output_dim == 0 else self.memory_output_dim
        self.linear: nn.Linear = nn.Linear(everything_dim, mem_out_dim)
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(mem_out_dim)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input_symbol, action, memory):
        """
        Update the memory given the latest input and action/output.

        :param input_symbol: represents the input at time t
        :param action: represents the output at time t
        :param memory: represents the memory at time t
        :return: the memory at time t + 1
        """
        if self.input_embedding_dim > 0:
            input_tensor = self.emb_linear(input_symbol)
            input_tensor = self.emb_norm(input_tensor)
            input_tensor = self.emb_activation_function(input_tensor)
        else:
            input_tensor = input_symbol

        input_tensor = torch.cat((input_tensor, memory), dim=1)
        if self.target_dim > 0:
            input_tensor = torch.cat((input_tensor, action), dim=1)
        if self.intermediate_layers > 0:
            linear = self.layers[0]
            norm = self.layers[1]
            relu = self.layers[2]
            intermediate = linear(input_tensor)
            intermediate = norm(intermediate)
            intermediate = relu(intermediate)
            for i in range(1, len(self.layers) // 3):
                intermediate = torch.cat((input_tensor, intermediate), dim=1)
                linear = self.layers[3 * i]
                norm = self.layers[3 * i + 1]
                relu = self.layers[3 * i + 2]
                intermediate = linear(intermediate)
                intermediate = norm(intermediate)
                intermediate = relu(intermediate)
            intermediate = torch.cat((input_tensor, intermediate), dim=1)
            final = self.linear(intermediate)
        else:
            final = self.linear(input_tensor)
        final = self.norm(final)
        sigmoid_outputs = self.sigmoid(final)

        return sigmoid_outputs


class ManyMemoryNN(nn.Module):
    """
    TODO comments
    """

    def __init__(self, num_mem_net: int, input_dim: int, memory_dim: int, input_embedding_dim: int = 0,
                 intermediate_layers: int = 0, intermediate_dim: int = 0, target_dim: int = 0,
                 funnel_to_zero_chunk: bool = False, activation_fn_is_tanh: bool = False,
                 emb_activation_fn_is_tanh: bool = False):
        """
        TODO comments
        :param num_mem_net:
        :param input_dim:
        :param memory_dim:
        :param input_embedding_dim:
        :param intermediate_layers:
        :param intermediate_dim:
        :param target_dim:
        """
        super().__init__()
        if num_mem_net < 2:
            raise ValueError("Must request at least 2 memory networks")
        self.num_mem_net: int = num_mem_net
        self.memory_chunk_dim: int = memory_dim // self.num_mem_net
        if self.memory_chunk_dim * self.num_mem_net != memory_dim:
            raise ValueError("Cannot integer divide memory into equal chunks.")
        self.zero_chunk_memory_dim: int = self.memory_chunk_dim
        self.funnel_to_zero_chunk: bool = funnel_to_zero_chunk
        if self.funnel_to_zero_chunk:
            self.zero_chunk_memory_dim = memory_dim
        self.memory_nets: nn.ModuleList = nn.ModuleList()
        for i in range(num_mem_net):
            self.memory_nets.append(
                MemoryNN(
                    input_dim, self.zero_chunk_memory_dim if i == 0 else self.memory_chunk_dim * 2,
                    input_embedding_dim=input_embedding_dim,
                    intermediate_layers=intermediate_layers,
                    intermediate_dim=intermediate_dim,
                    target_dim=target_dim,
                    memory_output_dim=self.memory_chunk_dim,
                    activation_fn_is_tanh=activation_fn_is_tanh,
                    emb_activation_fn_is_tanh=emb_activation_fn_is_tanh
                )
            )

    def forward(self, input_symbol, action, memory):
        """
        Update the memory given the latest input and action/output.

        :param input_symbol: represents the input at time t
        :param action: represents the output at time t
        :param memory: represents the memory at time t
        :return: the memory at time t + 1
        """
        memories = torch.chunk(memory, self.num_mem_net, dim=1)
        next_memory_chunks: list[torch.Tensor] = []
        for i in range(self.num_mem_net):
            memNet = self.memory_nets[i]
            if i == 0:
                memory_chunk = memory if self.funnel_to_zero_chunk else memories[i]
            else:
                memory_chunk = torch.cat((memories[0], memories[i]), dim=1)
            next_memory_chunk = memNet(input_symbol, action, memory_chunk)
            next_memory_chunks.append(next_memory_chunk)
        next_memory = torch.cat(next_memory_chunks, dim=1)
        return next_memory
