/*
 * GPU Packet Processing - Improved Inference Ring Buffer
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Zero-Copy Ring Buffer Implementation
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ring_buffer.h"
#include <doca_gpunetio.h>

/* Global variables: host and device pointer mapping */
static struct inference_ring_buffer *g_ring_host = NULL;
static struct inference_ring_buffer *g_ring_device = NULL;

/* Semaphore handle for CPU to GPU notification */
static struct doca_gpu_semaphore *g_sem_inference_cpu = NULL;

/* GPU kernel to get current GPU clock64() value */
__global__ void get_gpu_clock_kernel(uint64_t *output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = clock64();
    }
}

struct inference_ring_buffer* init_inference_ring_buffer(int gpu_id)
{
    cudaSetDevice(gpu_id);

    /* cudaHostAllocMapped: CPU and GPU share the same memory */
    if (cudaHostAlloc(&g_ring_host, sizeof(*g_ring_host),
                      cudaHostAllocMapped) != cudaSuccess) {
        return NULL;
    }

    /* Get GPU-side pointer */
    if (cudaHostGetDevicePointer(&g_ring_device, g_ring_host, 0) != cudaSuccess) {
        cudaFreeHost(g_ring_host);
        return NULL;
    }

    /* Initialize slots */
    for (int i = 0; i < INFERENCE_RING_SIZE; i++) {
        g_ring_host->slots[i].len = 0;
        g_ring_host->slots[i].ready = UVM_STATUS_FREE;
        g_ring_host->slots[i].request_id = 0;
        memset(g_ring_host->slots[i].data, 0, sizeof(g_ring_host->slots[i].data));
    }

    g_ring_host->head = 0;
    g_ring_host->tail = 0;
    g_ring_host->next_request_id = 1;
    g_ring_host->batch_epoch = 0;
    g_ring_host->pending_count = 0;

    /* Clock synchronization - measure actual GPU frequency */
    uint64_t *d_gpu_clock;
    cudaMalloc(&d_gpu_clock, sizeof(uint64_t));

    struct timespec ts1, ts2;
    uint64_t gpu_clock1, gpu_clock2;

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    get_gpu_clock_kernel<<<1, 1>>>(d_gpu_clock);
    cudaDeviceSynchronize();
    cudaMemcpy(&gpu_clock1, d_gpu_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    usleep(100000);  /* Wait 100ms to measure frequency */

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    get_gpu_clock_kernel<<<1, 1>>>(d_gpu_clock);
    cudaDeviceSynchronize();
    cudaMemcpy(&gpu_clock2, d_gpu_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    /* Calculate actual frequency */
    uint64_t cpu_ns1 = (uint64_t)ts1.tv_sec * 1000000000ULL + (uint64_t)ts1.tv_nsec;
    uint64_t cpu_ns2 = (uint64_t)ts2.tv_sec * 1000000000ULL + (uint64_t)ts2.tv_nsec;
    uint64_t gpu_cycles_elapsed = gpu_clock2 - gpu_clock1;
    uint64_t cpu_ns_elapsed = cpu_ns2 - cpu_ns1;
    double gpu_freq_ghz = (double)gpu_cycles_elapsed / (double)cpu_ns_elapsed;

    /* Use first measurement as baseline */
    uint64_t gpu_clock_base = gpu_clock1;
    uint64_t cpu_time_ns_base = cpu_ns1;

    cudaFree(d_gpu_clock);

    /* Save to ring buffer */
    g_ring_host->clock_sync.gpu_clock_base = gpu_clock_base;
    g_ring_host->clock_sync.cpu_time_ns_base = cpu_time_ns_base;
    g_ring_host->clock_sync.gpu_clock_freq_ghz = gpu_freq_ghz;

    /* Ensure CPU writes are visible to GPU */
    cudaDeviceSynchronize();

    return g_ring_device;  /* GPU kernels use this */
}

void free_inference_ring_buffer(struct inference_ring_buffer *ring)
{
    if (g_ring_host) {
        cudaFreeHost(g_ring_host);
        g_ring_host = NULL;
        g_ring_device = NULL;
    }
    g_sem_inference_cpu = NULL;
}

/* Set inference result notification semaphore (called after semaphore creation) */
void set_inference_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu)
{
    g_sem_inference_cpu = sem_cpu;
}

/* Set request notification semaphore (GPU to CPU) - no-op, kept for API compatibility */
void set_request_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu)
{
    (void)sem_cpu;  /* Unused - batch read function removed */
}

/* CPU write - UVM with explicit CUDA sync for concurrency */
int cpu_write_inference_result_to_gpu_ring(struct inference_ring_buffer *ring_gpu,
                                            uint32_t slot_index,
                                            const char *result)
{
    if (slot_index >= INFERENCE_RING_SIZE) {
        return 0;
    }

    struct inference_ring_slot *slot = &g_ring_host->slots[slot_index];

    /* Write inference result data */
    uint32_t len = 0;
    while (result[len] != '\0' && len < 895) {
        slot->data[len] = result[len];
        len++;
    }
    slot->data[len] = '\0';
    slot->len = len;

    /* Full memory barrier before updating status */
    __sync_synchronize();

    /* Atomically update status */
    __atomic_store_n(&slot->ready, UVM_STATUS_RESULT_READY, __ATOMIC_RELEASE);

    /* Notify GPU via DOCA semaphore */
    if (g_sem_inference_cpu) {
        doca_gpu_semaphore_set_status(g_sem_inference_cpu, slot_index,
                                      DOCA_GPU_SEMAPHORE_STATUS_READY);
    }

    return 1;
}

/* GPU-side function: allocate ring slot with concurrency safety */
__device__ int gpu_alloc_ring_slot(struct inference_ring_buffer *ring, uint64_t *request_id)
{
    /* Try up to 64 times (RING_SIZE / 2) to find a free slot */
    for (int attempt = 0; attempt < 64; attempt++) {
        uint32_t my_head = atomicAdd((uint32_t*)&ring->head, 1);
        __threadfence_system();
        uint32_t index = my_head % INFERENCE_RING_SIZE;

        /* Atomically check and claim slot: FREE -> PROCESSING */
        uint32_t expected = UVM_STATUS_FREE;
        uint32_t old_status = atomicCAS((uint32_t*)&ring->slots[index].ready,
                                        expected, UVM_STATUS_PROCESSING);

        if (old_status == UVM_STATUS_FREE) {
            /* Successfully claimed slot */
            *request_id = atomicAdd((unsigned long long*)&ring->next_request_id, 1);
            ring->slots[index].request_id = *request_id;
            ring->slots[index].len = 0;

            /* T1: Slot allocated */
            ring->slots[index].t1_slot_allocated = clock64();

            return (int)index;
        }

        /* Slot occupied, try next one */
    }

    /* Ring buffer full after 64 attempts */
    return -1;
}

__device__ void gpu_store_inference_data_to_slot(struct inference_ring_buffer *ring, int slot_index, const char *data)
{
    struct inference_ring_slot *slot = &ring->slots[slot_index];

    /* Record request start timestamp for end-to-end measurement */
    slot->start_timestamp = clock64();

    uint32_t len = 0;
    while (data[len] != '\0' && len < 895) {
        slot->data[len] = data[len];
        len++;
    }
    slot->data[len] = '\0';
    slot->len = len;

    __threadfence_system();

    /* T2: GPU wrote to UVM */
    slot->t2_gpu_wrote_uvm = clock64();

    slot->ready = UVM_STATUS_PARAM_READY;
}

__device__ int gpu_read_inference_result_from_slot(struct inference_ring_buffer *ring, int slot_index, char *output)
{
    struct inference_ring_slot *slot = &ring->slots[slot_index];

    /* Poll with fence on every iteration for CPU write visibility */
    int max_polls = 20000;

    for (int poll = 0; poll < max_polls; poll++) {
        /* Fence every poll to ensure CPU writes are visible */
        __threadfence_system();

        /* Volatile read of status */
        uint32_t current_status = slot->ready;

        if (current_status == UVM_STATUS_RESULT_READY) {
            /* T7: GPU detected result ready */
            slot->t7_gpu_read = clock64();

            uint32_t len = slot->len;
            for (uint32_t i = 0; i < len && i < 895; i++) {
                output[i] = slot->data[i];
            }
            output[len] = '\0';

            /* Reset to FREE for slot reuse */
            __threadfence_system();
            atomicExch((unsigned int*)&slot->ready, UVM_STATUS_FREE);

            return 1;
        }

        /* Yield every 200 polls to give CPU execution time */
        if (poll % 200 == 0 && poll > 0) {
            __nanosleep(500);  /* 0.5us */
        }
    }

    /* Timeout - mark slot as FREE for reuse */
    atomicExch((unsigned int*)&slot->ready, UVM_STATUS_FREE);
    return 0;
}
