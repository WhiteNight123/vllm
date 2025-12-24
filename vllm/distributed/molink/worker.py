# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoLink Worker for cross-node pipeline parallelism.

This module provides utilities for workers in a MoLink distributed pipeline.
In MoLink, each physical node runs with local pp_size=1, but the model layers
are explicitly distributed based on the MoLink configuration.

Key design principles:
1. Workers use the standard GPU Worker implementation
2. Layer distribution is controlled by MoLink parallel state
3. The MoLink parallel state overrides get_pp_indices for layer selection
"""

from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class MolinkWorkerMixin:
    """Mixin class for MoLink worker functionality.
    
    This mixin provides methods for handling intermediate tensors
    received via gRPC in a MoLink distributed pipeline.
    """
    
    def __init__(self):
        # Cache for intermediate tensors received via gRPC
        self._grpc_intermediate_tensors: Optional[IntermediateTensors] = None
        self._output_intermediate_tensors: Optional[IntermediateTensors] = None
    
    def set_intermediate_tensors(
        self, 
        intermediate_tensors: IntermediateTensors
    ) -> None:
        """Set intermediate tensors received from gRPC.
        
        This is called by the executor when intermediate tensors are
        received from the previous pipeline stage via gRPC.
        
        Args:
            intermediate_tensors: The intermediate tensors from previous stage.
        """
        self._grpc_intermediate_tensors = intermediate_tensors
    
    def get_grpc_intermediate_tensors(self) -> Optional[IntermediateTensors]:
        """Get and clear cached gRPC intermediate tensors.
        
        Returns:
            The cached intermediate tensors, or None.
        """
        tensors = self._grpc_intermediate_tensors
        self._grpc_intermediate_tensors = None
        return tensors
    
    def set_output_intermediate_tensors(
        self,
        intermediate_tensors: IntermediateTensors
    ) -> None:
        """Set output intermediate tensors for gRPC transmission.
        
        Args:
            intermediate_tensors: The intermediate tensors to send.
        """
        self._output_intermediate_tensors = intermediate_tensors
    
    def get_output_intermediate_tensors(self) -> Optional[IntermediateTensors]:
        """Get and clear output intermediate tensors.
        
        Returns:
            The output intermediate tensors, or None.
        """
        tensors = self._output_intermediate_tensors
        self._output_intermediate_tensors = None
        return tensors


def init_molink_for_worker(
    vllm_config: VllmConfig,
    rank: int,
    local_rank: int,
) -> None:
    """Initialize MoLink parallel state in worker process.
    
    This should be called during worker initialization to set up
    the MoLink parallel state which controls layer distribution.
    
    Args:
        vllm_config: The vLLM configuration.
        rank: The global rank.
        local_rank: The local rank.
    """
    from .parallel_state import init_molink_parallel_state, apply_molink_patches
    
    molink_config = getattr(vllm_config, 'molink_config', None)
    
    if molink_config is not None and molink_config.enabled:
        start_layer, end_layer = molink_config.get_serving_layers()
        init_molink_parallel_state(
            enabled=True,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        # Apply patches after initialization
        apply_molink_patches()
        
        logger.info(
            f"Worker {rank} (local {local_rank}): MoLink initialized "
            f"with layers {start_layer}-{end_layer}"
        )
