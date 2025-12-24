# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration for MoLink cross-node pipeline parallelism.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MolinkConfig:
    """Configuration for MoLink cross-node pipeline parallelism.
    
    MoLink enables distributed pipeline parallelism across multiple physical
    nodes using gRPC for communication. Each node can run a subset of the
    model layers, enabling inference of very large models that don't fit
    on a single node.
    
    Attributes:
        enabled: Whether MoLink is enabled.
        initial_peer: The gRPC address of the initial node to join the pipeline.
                     If None, this node is the head node.
        grpc_port: The gRPC port to listen on. If 0, a free port will be chosen.
        start_layer: The first layer this node handles (inclusive).
        end_layer: The last layer this node handles (exclusive).
        max_message_size_mb: Maximum gRPC message size in MB.
        connection_timeout_s: Timeout for gRPC connections in seconds.
        heartbeat_interval_s: Interval for health check heartbeats.
        enable_compression: Whether to enable gRPC message compression.
        num_delivery_workers: Number of workers for async tensor delivery.
    """
    enabled: bool = False
    initial_peer: Optional[str] = None
    grpc_port: int = 0  # 0 means auto-select
    start_layer: int = 0
    end_layer: int = -1  # -1 means all remaining layers
    max_message_size_mb: int = 200
    connection_timeout_s: float = 30.0
    heartbeat_interval_s: float = 5.0
    enable_compression: bool = False
    num_delivery_workers: int = 2
    
    def __post_init__(self):
        if self.max_message_size_mb <= 0:
            raise ValueError("max_message_size_mb must be positive")
        if self.connection_timeout_s <= 0:
            raise ValueError("connection_timeout_s must be positive")
        if self.num_delivery_workers <= 0:
            raise ValueError("num_delivery_workers must be positive")
        
    @property
    def is_head_node(self) -> bool:
        """Returns True if this is the head node of the pipeline."""
        return self.initial_peer is None or self.initial_peer == ''
    
    @property
    def max_message_size_bytes(self) -> int:
        """Returns the maximum message size in bytes."""
        return self.max_message_size_mb * 1024 * 1024
    
    def get_serving_layers(self) -> tuple[int, int]:
        """Returns the range of layers this node serves."""
        return (self.start_layer, self.end_layer)
