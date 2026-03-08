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
- DOCA SDK 2.9+
- CUDA 12.6+
- TensorRT 10.4 (last version supporting SM 7.0 / V100)
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
#   -e  Path to TensorRT engine file
```

## HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/index.` | GET | Static index page |
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

## License

BSD-3-Clause. See individual source files for details.

## Acknowledgments

Built on NVIDIA DOCA SDK and GPUNetIO technology.
