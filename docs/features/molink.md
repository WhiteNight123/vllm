# MoLink: Cross-Node Pipeline Parallelism for vLLM

MoLink enables distributed pipeline parallelism across multiple physical nodes using gRPC for communication. This allows running very large models (e.g., 70B+ parameters) that don't fit on a single node by distributing model layers across multiple nodes.

## Overview

MoLink extends vLLM's pipeline parallelism to work across physical nodes. Unlike traditional PP which uses NCCL for communication within a node, MoLink uses gRPC to transfer intermediate tensors between nodes.

### Architecture

```
   Node 1 (Head)                    Node 2 (Worker)
   Layers 0-15                      Layers 16-31
   +-----------------+              +-----------------+
   |   EngineCore    |              |   EngineCore    |
   |       |         |              |       |         |
   | +-------------+ |              | +-------------+ |
   | |MolinkExecutor|<--gRPC------->|MolinkExecutor | |
   | +-------------+ |              | +-------------+ |
   |       |         |              |       |         |
   | +-------------+ |              | +-------------+ |
   | |   Workers   | |              | |   Workers   | |
   | +-------------+ |              | +-------------+ |
   +-----------------+              +-----------------+
```

### Execution Flow

1. **Request arrives** at the head node
2. **Head node** schedules the batch and executes its layers (0-15)
3. **Intermediate tensors** are sent to the worker node via gRPC
4. **Worker node** receives tensors and executes its layers (16-31)
5. **Final output** is sent back to the head node via gRPC
6. **Head node** returns the result to the client

## Configuration

### MolinkConfig

```python
from vllm.config.molink import MolinkConfig

config = MolinkConfig(
    enabled=True,                    # Enable MoLink
    initial_peer=None,               # None for head node, or "IP:port" for workers
    grpc_port=0,                     # gRPC port (0 for auto)
    start_layer=0,                   # First layer this node handles
    end_layer=16,                    # Last layer this node handles
    max_message_size_mb=200,         # Max gRPC message size
    connection_timeout_s=30.0,       # Connection timeout
    heartbeat_interval_s=5.0,        # Health check interval
)
```

## Usage

### Two-Node Setup Example

Suppose you want to run Qwen3-14B across two nodes, each with 2 GPUs:

**Node 1 (Head - IP: 192.168.1.100):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --tensor-parallel-size 2 \
    --distributed-executor-backend molink \
    --molink-config '{"enabled": true, "start_layer": 0, "end_layer": 16}'
```

**Node 2 (Worker - IP: 192.168.1.101):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --tensor-parallel-size 2 \
    --distributed-executor-backend molink \
    --molink-config '{"enabled": true, "initial_peer": "192.168.1.100:50051", "start_layer": 16, "end_layer": 32}'
```

### Programmatic Usage

```python
from vllm import LLM, SamplingParams
from vllm.config.molink import MolinkConfig

# Head node configuration
molink_config = MolinkConfig(
    enabled=True,
    start_layer=0,
    end_layer=16,
)

llm = LLM(
    model="Qwen/Qwen3-14B",
    tensor_parallel_size=2,
    molink_config=molink_config,
)

# Generate as usual
outputs = llm.generate(
    ["The future of AI is"],
    SamplingParams(max_tokens=100)
)
```

## Components

### MolinkExecutor

The `MolinkExecutor` extends `MultiprocExecutor` to handle cross-node communication:

- Manages gRPC server for receiving tensors
- Coordinates execution across pipeline stages
- Handles tensor serialization and delivery

### MolinkService

gRPC service implementing:

- `JoinPipeline`: Register new worker nodes
- `PushIntermediateTensors`: Receive tensors from previous stage
- `PushSamplerOutput`: Receive final output
- `ExecuteWorkerStep`: Trigger execution on worker nodes

### TensorDeliveryManager

Background process for async tensor delivery:

- Serializes tensors to bytes
- Sends via gRPC without blocking main computation
- Enables overlapping of communication and computation

### MolinkWorker

Extended GPU worker supporting:

- Receiving intermediate tensors via gRPC
- Forwarding results to next stage or back to head

## Performance Considerations

1. **Network Bandwidth**: Cross-node PP requires significant bandwidth for tensor transfer. Use high-speed networking (100Gbps+) for best performance.

2. **Layer Distribution**: Distribute layers evenly across nodes to minimize pipeline bubbles.

3. **Batch Size**: Larger batches amortize communication overhead.

4. **Async Delivery**: The `TensorDeliveryManager` runs in a separate process to overlap communication with computation.

## Limitations

1. **Latency**: Cross-node communication adds latency compared to single-node PP.

2. **Network Dependency**: Performance depends on network quality and bandwidth.

3. **Fault Tolerance**: Currently no automatic recovery if a node fails.

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure firewall allows gRPC port (default: 50051+)

2. **Timeout errors**: Increase `connection_timeout_s` for slow networks

3. **Out of memory**: Reduce `max_message_size_mb` if hitting memory limits

### Debugging

Enable debug logging:
```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

Check gRPC connectivity:
```python
from vllm.distributed.molink.utils import extract_ip, find_free_port
print(f"This node IP: {extract_ip()}")
print(f"Available port: {find_free_port(50051)}")
```

## Future Work

- [ ] Automatic layer distribution based on model architecture
- [ ] Support for more than 2 nodes
- [ ] Fault tolerance and automatic recovery
- [ ] Compression for tensor transfer
- [ ] Profiling and optimization tools
