#!/bin/bash

# GPUNet - Build and Run Script
# Automates environment setup, compilation and execution

set -e

echo "======================================================="
echo "GPUNet - Automated Build and Run Script"
echo "======================================================="

# 1. Set environment variables
echo "Setting environment variables..."
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/opt/mellanox/doca/lib/x86_64-linux-gnu/pkgconfig:/opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig:/opt/mellanox/grpc/lib/pkgconfig:/opt/mellanox/flexio/lib/pkgconfig
export PATH=${PATH}:/opt/mellanox/doca/tools:/opt/mellanox/grpc/bin
export PATH="/usr/local/cuda/bin:${PATH}"
export CPATH="$(echo /usr/local/cuda/targets/{x86_64,sbsa}-linux/include | sed 's/ /:/'):${CPATH}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/mellanox/gdrcopy/src
export LD_LIBRARY_PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 2. Load required kernel modules
echo "Loading nvidia-peermem module..."
sudo modprobe nvidia-peermem

echo "Checking module status:"
lsmod | egrep "gdrdrv|nvidia"

# 3. Clean old build directory and rebuild
echo "Cleaning old build directory..."
sudo rm -rf /tmp/build-gpunet

echo "Configuring meson build..."
cd /opt/mellanox/doca/applications
# Create symlink to source
sudo rm -f gpu_packet_processing
sudo ln -sf /home/lsalab/gpu_packet_processing_v2_github gpu_packet_processing
sudo env PATH="/usr/local/cuda/bin:${PATH}" \
     PKG_CONFIG_PATH="${PKG_CONFIG_PATH}" \
     CPATH="${CPATH}" \
     meson /tmp/build-gpunet -Denable_all_applications=false -Denable_gpu_packet_processing=true

# 4. Compile
echo "Compiling GPUNet..."
sudo env PATH="/usr/local/cuda/bin:${PATH}" ninja -C /tmp/build-gpunet

# Clean up symlink
sudo rm -f gpu_packet_processing

echo "======================================================="
echo "Build complete! Executable: /tmp/build-gpunet/gpu_packet_processing/gpunet"
echo "======================================================="

# 5. Ask if user wants to run immediately
read -p "Run GPUNet now? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running GPUNet..."
    echo "Note: This is an infinite loop program!"

    read -p "Enter run time in seconds [default 10]: " -r RUN_TIME
    RUN_TIME=${RUN_TIME:-10}

    echo "Program will run for ${RUN_TIME} seconds then stop..."
    echo "You can also use Ctrl+C to stop manually"
    echo ""

    echo "Running in background for ${RUN_TIME} seconds..."
    sudo /tmp/build-gpunet/gpu_packet_processing/gpunet -n c1:00.0 -g e6:00.0 -q 4 -s &

    sleep ${RUN_TIME}

    echo ""
    echo "Stopping program..."
    GPUNET_PIDS=$(pgrep -f "gpunet.*c1:00.0.*e6:00.0")
    if [ -n "$GPUNET_PIDS" ]; then
        echo "Found PIDs: $GPUNET_PIDS"
        for pid in $GPUNET_PIDS; do
            echo "Stopping PID $pid..."
            sudo kill -KILL $pid 2>/dev/null || echo "PID $pid may have stopped"
        done
    else
        echo "No running process found"
    fi

    echo "Program stopped after ${RUN_TIME} seconds"
fi

echo "Script completed!"
