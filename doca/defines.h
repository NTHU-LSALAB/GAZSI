/*
 * GPU Packet Processing - Definitions
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GPU_PKT_DEFINES_H
#define GPU_PKT_DEFINES_H

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <utils.h>
#include <signal.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_version.h>
#include <doca_log.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_dev.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_mmap.h>
#include <doca_argp.h>
#include <doca_dpdk.h>
#include <doca_flow.h>
#include <doca_pe.h>

/* GPU page size */
#define GPU_PAGE_SIZE (1UL << 16)
#define WARP_SIZE 32
#define WARP_FULL_MASK 0xFFFFFFFF
#define MAX_PORT_STR_LEN 128 /* Maximal length of port name */
#define MAX_QUEUES 4
#define MAX_PKT_NUM 65536
#define MAX_PKT_SIZE 8192
#define MAX_RX_NUM_PKTS_HTTP 64
#define MAX_RX_TIMEOUT_NS_HTTP 0 /* Busy-polling for zero-copy GPU-Direct (zero packet loss) */
#define MAX_SQ_DESCR_NUM 4096
#define CUDA_THREADS 512
#define ETHER_ADDR_LEN 6
#define BYTE_SWAP16(v) ((((uint16_t)(v)&UINT16_C(0x00ff)) << 8) | (((uint16_t)(v)&UINT16_C(0xff00)) >> 8))

#define BYTE_SWAP32(x) \
	((((x)&0xff000000) >> 24) | (((x)&0x00ff0000) >> 8) | (((x)&0x0000ff00) << 8) | (((x)&0x000000ff) << 24))

/* Each thread in the HTTP server CUDA kernel warp has its own subset of 32 buffers */
#define TX_BUF_NUM 1024 /* 32 x 32 */
/* TX buffer size: 2048 bytes to accommodate large HTTP responses (128+ tokens) */
#define TX_BUF_MAX_SZ 2048
#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
/* DPDK port to accept new TCP connections */
#define DPDK_DEFAULT_PORT 0
/* HTTP page type to send as response to HTTP GET request */
enum http_page_get {
	HTTP_GET_INDEX = 0, /* HTML index page */
	HTTP_GET_CONTACTS,  /* HTML contact page */
	HTTP_GET_INFERENCE, /* Inference request with parameters */
	HTTP_GET_NOT_FOUND  /* HTML not found page */
};

#endif /* GPU_PKT_DEFINES_H */
