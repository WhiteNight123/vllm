# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoLink: Cross-Node Pipeline Parallelism for vLLM v1.

MoLink enables distributed pipeline parallelism across multiple physical
nodes using gRPC for communication. Each node can run a subset of the
model layers, enabling inference of very large models that don't fit
on a single node.

Architecture Overview:
- Each physical node runs with local pp_size=1 (no NCCL PP within node)
- Model layers are explicitly distributed based on MoLink configuration
- Cross-node communication is handled entirely via gRPC
- The head node orchestrates execution across all nodes

Key Components:
- MolinkExecutor: Extends MultiprocExecutor for cross-node PP
- MolinkService: gRPC service for handling inter-node communication
- TensorDeliveryManager: Async tensor delivery in background process
- parallel_state: MoLink-aware layer distribution utilities

Usage:
1. Start head node:
   --distributed-executor-backend molink
   --molink-enabled
   --molink-grpc-port 50051
   --molink-start-layer 0
   --molink-end-layer 20

2. Start worker node:
   --distributed-executor-backend molink
   --molink-enabled
   --molink-initial-peer "192.168.1.100:50051"
   --molink-start-layer 20
   --molink-end-layer -1
"""

from .executor import MolinkExecutor
from .service import MolinkService
from .delivery import TensorDeliveryManager, TensorDeliveryProcess
from .parallel_state import (
    init_molink_parallel_state,
    destroy_molink_parallel_state,
    is_molink_enabled,
    get_molink_layer_range,
    get_molink_pp_indices,
    is_molink_first_stage,
    is_molink_last_stage,
    get_molink_pp_rank,
    get_molink_pp_size,
    update_molink_stage_info,
    patch_get_pp_indices,
    patch_pp_group,
    apply_molink_patches,
)
from .utils import (
    extract_ip,
    find_free_port,
    get_grpc_options,
    PipelineTopology,
    NodeInfo,
    serialize_metadata,
    deserialize_metadata,
)
from .worker import (
    MolinkWorkerMixin,
    init_molink_for_worker,
)

__all__ = [
    # Executor
    "MolinkExecutor",
    
    # Service
    "MolinkService",
    
    # Delivery
    "TensorDeliveryManager",
    "TensorDeliveryProcess",
    
    # Parallel State
    "init_molink_parallel_state",
    "destroy_molink_parallel_state",
    "is_molink_enabled",
    "get_molink_layer_range",
    "get_molink_pp_indices",
    "is_molink_first_stage",
    "is_molink_last_stage",
    "get_molink_pp_rank",
    "get_molink_pp_size",
    "update_molink_stage_info",
    "patch_get_pp_indices",
    "patch_pp_group",
    "apply_molink_patches",
    
    # Utils
    "extract_ip",
    "find_free_port",
    "get_grpc_options",
    "PipelineTopology",
    "NodeInfo",
    "serialize_metadata",
    "deserialize_metadata",
    
    # Worker
    "MolinkWorkerMixin",
    "init_molink_for_worker",
]
