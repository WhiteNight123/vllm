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
    """

 