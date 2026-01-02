/*
 * GPU Packet Processing - Improved Inference Ring Buffer Header
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Lock-free ring buffer for CPU-GPU inference data exchange
 */

#ifndef IMPROVED_INFERENCE_UVM_H
#define IMPROVED_INFERENCE_UVM_H

#include <stdint.h>

/* Ring Buffer size - 128 performs better than 256 in testing */
#define INFERENCE_RING_SIZE 128

/* UVM buffer states - Producer-Consumer state machine */
#define UVM_STATUS_FREE         0  /* Buffer is idle */
#define UVM_STATUS_PARAM_READY  1  /* GPU has written parameters */
#define UVM_STATUS_PROCESSING   2  /* CPU is processing (atomic claim) */
#define UVM_STATUS_RESULT_READY 3  /* CPU has written inference result */
#define UVM_STATUS_CONSUMED     4  /* GPU has read result */

/*
 * Ring Buffer slot structure
 * Cache-line aligned: hot data in first cache line (128 bytes)
 */
struct inference_ring_slot {
    /* Cache Line 1: Hot data - frequently accessed fields (128 bytes) */
    volatile uint32_t ready;           /* State - polled by GPU/CPU */
    volatile uint32_t len;             /* Current data length */
    volatile uint64_t request_id;      /* Request ID */
    volatile uint64_t start_timestamp; /* GPU processing start timestamp */

    /* Performance timing (9 timestamps for detailed profiling) */
    volatile uint64_t t0_gpu_received;   /* T0: HTTP request arrived at GPU */
    volatile uint64_t t1_slot_allocated; /* T1: GPU allocated ring slot */
    volatile uint64_t t2_gpu_wrote_uvm;  /* T2: GPU finished writing to UVM */
    volatile uint64_t t3_cpu_read;       /* T3: CPU detected data */
    volatile uint64_t t4_tensorrt_start; /* T4: CPU started TensorRT inference */
    volatile uint64_t t5_tensorrt_end;   /* T5: TensorRT inference complete */
    volatile uint64_t t6_cpu_wrote_uvm;  /* T6: CPU wrote result to UVM */
    volatile uint64_t t7_gpu_read;       /* T7: GPU detected result */
    volatile uint64_t t8_gpu_sent;       /* T8: GPU sent HTTP response */

    char padding1[16];                 /* Padding to 128 bytes */

    /* TCP connection info for HTTP response construction */
    uint8_t eth_src_addr_bytes[6];     /* Ethernet source address */
    uint8_t eth_dst_addr_bytes[6];     /* Ethernet destination address */
    uint32_t ip_src_addr;              /* IP source address */
    uint32_t ip_dst_addr;              /* IP destination address */
    uint16_t ip_total_length;          /* IP total length */
    uint16_t tcp_src_port;             /* TCP source port */
    uint16_t tcp_dst_port;             /* TCP destination port */
    uint8_t tcp_dt_off;                /* TCP data offset */
    uint8_t http_page_type;            /* HTTP page type (enum http_page_get) */
    uint32_t tcp_sent_seq;             /* TCP sent sequence number */
    uint32_t tcp_recv_ack;             /* TCP receive acknowledgment */
    char padding_tcp[2];               /* Alignment padding */
    /* TCP info: 6+6+4+4+2+2+2+1+1+4+4+2 = 38 bytes, aligned to 40 */

    /* Cache Lines 2-8: Cold data - accessed only when needed */
    char data[856];                    /* Data buffer (896 - 40 = 856) */
    char padding2[128];                /* Align to cache lines */
} __attribute__((aligned(128)));

/*
 * Clock sync info for GPU/CPU clock conversion
 */
struct clock_sync_info {
    uint64_t gpu_clock_base;      /* GPU clock64() baseline */
    uint64_t cpu_time_ns_base;    /* CPU clock_gettime() baseline (ns) */
    double   gpu_clock_freq_ghz;  /* GPU clock frequency (GHz) */
};

/*
 * Ring Buffer structure
 * Lock-free producer-consumer ring buffer
 */
struct inference_ring_buffer {
    struct inference_ring_slot slots[INFERENCE_RING_SIZE];
    volatile uint32_t head;    /* Producer (GPU) write position */
    volatile uint32_t tail;    /* Consumer (CPU) read position */
    volatile uint64_t next_request_id;

    /* GPU Batch coordination for warp synchronization */
    volatile uint32_t batch_epoch;        /* Incrementing epoch for batch completion */
    volatile uint32_t pending_count;      /* Current pending request count */
    char batch_padding[56];               /* Align to cache line */

    /* Clock sync - set once at init, read-only thereafter */
    struct clock_sync_info clock_sync;
};

/* Forward declaration for DOCA types */
struct doca_gpu_semaphore;
struct doca_gpu_semaphore_gpu;

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Ring Buffer. Returns pointer, NULL on failure */
struct inference_ring_buffer* init_inference_ring_buffer(int doca_gpu_id);

/* Set inference result notification semaphore (CPU-side, for notifying GPU after writing result) */
void set_inference_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu);

/* Set request notification semaphore (GPU -> CPU notification) */
void set_request_semaphore_cpu(struct doca_gpu_semaphore *sem_cpu);

/* Free Ring Buffer */
void free_inference_ring_buffer(struct inference_ring_buffer *ring);

/* CPU-side write processing result to GPU ring. Returns 1 on success, 0 on failure */
int cpu_write_inference_result_to_gpu_ring(struct inference_ring_buffer *ring_gpu, uint32_t slot_index, const char *result);

/* GPU-side slot allocation. Returns slot index, -1 if ring buffer is full */
#ifdef __CUDACC__
__device__ int gpu_alloc_ring_slot(struct inference_ring_buffer *ring, uint64_t *request_id);

/* GPU-side store inference data to slot */
__device__ void gpu_store_inference_data_to_slot(struct inference_ring_buffer *ring, int slot_index, const char *data);

/* GPU-side read result from slot. Returns 1 on success, 0 if result not ready */
__device__ int gpu_read_inference_result_from_slot(struct inference_ring_buffer *ring, int slot_index, char *output);
#endif

#ifdef __cplusplus
}
#endif

#endif /* IMPROVED_INFERENCE_UVM_H */
