# GAZSI: GPU-Accelerated Native Zero-copy for Socket-Based Inference Serving Framework

A high-performance GPU-accelerated HTTP server and packet processing framework using NVIDIA DOCA SDK and GPUNetIO.

## Overview

GAZSI enables direct GPU processing of network packets, bypassing the CPU for minimal latency. The system implements:

- **GPU-Direct HTTP Server**: HTTP request/response handling entirely on GPU
- **TensorRT Inference Integration**: ML inference via lock-free CPU-GPU ring buffer
- **Dynamic Batching**: Automatic request batching for improved throughput

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Network Interface                        │
│                  (ConnectX-6/7 SmartNIC)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ GPUNetIO
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        GPU Memory                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ TCP Receive │─►│ HTTP Parser  │─►│ Response Builder │   │
│  │   Kernel    │  │   Kernel     │  │     Kernel       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│         │                                                    │
│         ▼         Ring Buffer (UVM)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Inference Request Queue ◄──► CPU TensorRT Runner    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Zero-Copy Packet Processing**: Direct NIC-to-GPU data path
- **Lock-Free Ring Buffer**: Efficient CPU-GPU data exchange (128 slots)
- **Semaphore Signaling**: Event-driven CPU-GPU synchronization
- **Dynamic Batch Inference**: Up to 8 requests per batch
- **HTTP/1.1 Support**: GET/POST request handling on GPU

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.0+
- NVIDIA ConnectX-6 or newer SmartNIC
- 8GB+ System Memory

### Software
- Ubuntu 20.04/22.04
- DOCA SDK 2.0+
- CUDA 11.8+ / 12.x
- GCC 9+

## Building

The project builds as part of the DOCA SDK application framework:

```bash
# Create symlink in DOCA applications directory
sudo ln -s /path/to/GAZSI \
    /opt/mellanox/doca/applications/GAZSI

# Build
cd /opt/mellanox/doca/applications
sudo meson build -Denable_GAZSI=true
sudo ninja -C build
```

## Usage

```bash
# Basic usage
sudo ./GAZSI -g <GPU_PCI> -n <NIC_PCI> -q <NUM_QUEUES>

# With HTTP server
sudo ./GAZSI -g E6:00.0 -n c1:00.0 -q 2 -s

# Options:
#   -g  GPU PCIe address
#   -n  NIC PCIe address
#   -q  Number of receive queues (1-4)
#   -s  Enable HTTP server mode
```

## HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/index.` | GET | Static index page |
| `/contacts.` | GET | Static contacts page |
| `/inference?d=<text>` | GET | TensorRT inference endpoint |

### Inference Example

```bash
# Send inference request
curl "http://10.0.0.6/inference?d=hello%20world"

# Response
{
  "input": "hello world",
  "tokens": 3,
  "embedding_sample": [0.123, -0.456, 0.789],
  "inference_time_us": 1234,
  "batch_size": 1
}
```

## Project Structure

```
GAZSI/
├── main.c                      # Application entry point
├── kernels/
│   ├── tcp_rx.cu               # TCP packet receive kernel
│   ├── http_server.cu          # HTTP request/response handling
│   └── filters.cuh             # Packet filter functions
├── inference/
│   ├── tensorrt.cu             # TensorRT inference wrapper
│   ├── tensorrt.h              # TensorRT header
│   ├── ring_buffer.cu          # Lock-free ring buffer
│   └── ring_buffer.h           # Ring buffer header
├── doca/
│   ├── tcp.c                   # TCP queue initialization
│   ├── flow.c                  # DOCA Flow configuration
│   ├── common.h                # Common definitions
│   └── defines.h               # Constants and macros
├── tcp/
│   ├── session.c               # TCP session management
│   └── cpu_rss.c               # CPU RSS fallback
└── meson.build                 # Build configuration
```

## Performance

Typical latency breakdown for inference requests:

| Stage | Time |
|-------|------|
| T0→T1: Slot Allocation | ~10 us |
| T1→T2: GPU Write to UVM | ~5 us |
| T2→T3: CPU Detection | ~50 us |
| T3→T5: TensorRT Inference | ~500-2000 us |
| T5→T8: Response Sent | ~100 us |

## License

BSD-3-Clause. See individual source files for details.

## Acknowledgments

Built on NVIDIA DOCA SDK and GPUNetIO technology.
