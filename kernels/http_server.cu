/*
 * GPU Packet Processing - HTTP Server Kernel
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdlib.h>
#include <string.h>

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>

/* Global Atomic Counter for TX Buffer Allocation
 * Used for atomic allocation of TX buffer index across all warps to avoid race conditions
 */
__device__ unsigned long long g_tx_buf_counter = 0;

#include "common.h"
#include "packets.h"
#include "filters.cuh"
#include "ring_buffer.h"  /* Ring Buffer UVM */

/* HTTP Response Buffer Layout Constants */
#define HTTP_BODY_TEMP_OFFSET  256  /* Temporary offset for body construction */
#define HTTP_BODY_MAX_LEN      900  /* Max body length before truncation */

/* Pre-constructed HTTP Response Template for efficiency */
#define HTTP_RESP_PREFIX "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
#define HTTP_RESP_PREFIX_LEN 65  /* strlen(HTTP_RESP_PREFIX) - verified with Python */
#define HTTP_RESP_SUFFIX "\r\nConnection: keep-alive\r\n\r\n"
#define HTTP_RESP_SUFFIX_LEN 28  /* strlen(HTTP_RESP_SUFFIX) - verified with Python */

/* Warp-level parallel memcpy - each lane copies dst[lane_id], dst[lane_id+32], ... */
__device__ static inline void warp_memcpy(char *dst, const char *src, int len, int lane_id)
{
    for (int i = lane_id; i < len; i += 32) {
        dst[i] = src[i];
    }
}

/* Helper: Copy string to buffer, return bytes written */
__device__ static inline int strcpy_to_buf(char *dst, const char *src, int max_len)
{
	int len = 0;
	while (src[len] != '\0' && len < max_len)
		dst[len] = src[len], len++;
	return len;
}

/* Helper: Convert integer to string, return bytes written */
__device__ static inline int int_to_str(char *buf, int value)
{
	if (value == 0) {
		buf[0] = '0';
		return 1;
	}

	char digits[16];
	int num_digits = 0;
	while (value > 0) {
		digits[num_digits++] = '0' + (value % 10);
		value /= 10;
	}

	/* Reverse digits */
	for (int i = 0; i < num_digits; i++)
		buf[i] = digits[num_digits - 1 - i];

	return num_digits;
}

/* Global UVM buffer pointer */
__device__ struct inference_ring_buffer *g_inference_ring_buf = nullptr;

/* Global semaphore handle for CPU -> GPU notification */
__device__ struct doca_gpu_semaphore_gpu *g_sem_inference_gpu = nullptr;

/* Global semaphore handle for GPU -> CPU notification */
__device__ struct doca_gpu_semaphore_gpu *g_sem_request_gpu = nullptr;

/* Set Ring Buffer pointer - called from CPU */
__global__ void set_inference_ring_buffer(struct inference_ring_buffer *ring)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_inference_ring_buf = ring;
        printf("[RING_INIT] Ring Buffer set: %p\n", ring);
    }
}

/* Set inference semaphore GPU handle */
__global__ void set_inference_semaphore(struct doca_gpu_semaphore_gpu *sem)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_sem_inference_gpu = sem;
        printf("[SEM] Inference semaphore GPU handle set: %p\n", sem);
    }
}

/* Set request notification semaphore GPU handle */
__global__ void set_request_semaphore(struct doca_gpu_semaphore_gpu *sem)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_sem_request_gpu = sem;
        printf("[SEM] Request semaphore GPU handle set: %p\n", sem);
    }
}

/* GPU-side function to store inference data to Ring Buffer */
__device__ int store_inference_data_to_ring(const char* data, uint64_t *request_id)
{
    uint32_t lane_id = threadIdx.x % 32;

    if (g_inference_ring_buf == nullptr) {
        if (lane_id == 0) {
            printf("[RING_STORE] Error: Ring Buffer not initialized\n");
        }
        return -1;
    }

    /* TIMING: T0 - HTTP request arrived */
    uint64_t t0_timestamp = clock64();

    /* Allocate slot */
    int slot_index = gpu_alloc_ring_slot(g_inference_ring_buf, request_id);
    if (slot_index < 0) {
        if (lane_id == 0) {
            printf("[RING_STORE] Ring Buffer full, dropping request\n");
        }
        return -1;
    }

    /* Write T0 to slot */
    g_inference_ring_buf->slots[slot_index].t0_gpu_received = t0_timestamp;

    /* Store data to slot */
    gpu_store_inference_data_to_slot(g_inference_ring_buf, slot_index, data);

    /* GPU sets semaphore to notify CPU of new request */
    if (g_sem_request_gpu != nullptr && lane_id == 0) {
        __threadfence_system();  /* Ensure data visible to CPU */
        doca_gpu_dev_semaphore_set_status(g_sem_request_gpu, slot_index,
                                          DOCA_GPU_SEMAPHORE_STATUS_READY);
    }

    return slot_index;
}

/* GPU-side read result from Ring Buffer - uses DOCA semaphore wait instead of polling */
__device__ bool read_inference_result_from_ring(int slot_index, char *output)
{
    if (g_inference_ring_buf == nullptr || slot_index < 0) {
        return false;
    }

    if (g_sem_inference_gpu != nullptr) {
        enum doca_gpu_semaphore_status sem_status;
        int max_wait = 100000;

        for (int wait = 0; wait < max_wait; wait++) {
            doca_error_t ret = doca_gpu_dev_semaphore_get_status(
                g_sem_inference_gpu, slot_index, &sem_status);

            if (ret == DOCA_SUCCESS && sem_status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
                struct inference_ring_slot *slot = &g_inference_ring_buf->slots[slot_index];
                slot->t7_gpu_read = clock64();

                uint32_t len = slot->len;
                for (uint32_t i = 0; i < len && i < 895; i++) {
                    output[i] = slot->data[i];
                }
                output[len] = '\0';

                doca_gpu_dev_semaphore_set_status(g_sem_inference_gpu, slot_index,
                                                   DOCA_GPU_SEMAPHORE_STATUS_FREE);
                __threadfence_system();
                atomicExch((unsigned int*)&slot->ready, UVM_STATUS_FREE);

                return true;
            }

            if (wait % 1000 == 0 && wait > 0) {
                __nanosleep(100);
            }
        }

        printf("[SEM] Timeout for slot %d\n", slot_index);
        doca_gpu_dev_semaphore_set_status(g_sem_inference_gpu, slot_index,
                                           DOCA_GPU_SEMAPHORE_STATUS_FREE);
        atomicExch((unsigned int*)&g_inference_ring_buf->slots[slot_index].ready, UVM_STATUS_FREE);
        return false;
    }

    /* Fallback to polling */
    int success = gpu_read_inference_result_from_slot(g_inference_ring_buf, slot_index, output);
    return (success != 0);
}

DOCA_LOG_REGISTER(GPUNET::KernelHttpServer);

static
__device__ void http_set_mac_addr(struct eth_ip_tcp_hdr *hdr, const uint16_t *src_bytes, const uint16_t *dst_bytes)
{
	((uint16_t *)hdr->l2_hdr.s_addr_bytes)[0] = src_bytes[0];
	((uint16_t *)hdr->l2_hdr.s_addr_bytes)[1] = src_bytes[1];
	((uint16_t *)hdr->l2_hdr.s_addr_bytes)[2] = src_bytes[2];

	((uint16_t *)hdr->l2_hdr.d_addr_bytes)[0] = dst_bytes[0];
	((uint16_t *)hdr->l2_hdr.d_addr_bytes)[1] = dst_bytes[1];
	((uint16_t *)hdr->l2_hdr.d_addr_bytes)[2] = dst_bytes[2];
}

__global__ void cuda_kernel_http_server(uint32_t *exit_cond,
					struct doca_gpu_eth_txq *txq0, struct doca_gpu_eth_txq *txq1, struct doca_gpu_eth_txq *txq2, struct doca_gpu_eth_txq *txq3,
					int sem_num,
					struct doca_gpu_semaphore_gpu *sem_http0, struct doca_gpu_semaphore_gpu *sem_http1, struct doca_gpu_semaphore_gpu *sem_http2, struct doca_gpu_semaphore_gpu *sem_http3,
					struct doca_gpu_buf_arr *buf_arr_gpu_page_index, uint32_t nbytes_page_index,
					struct doca_gpu_buf_arr *buf_arr_gpu_page_contacts, uint32_t nbytes_page_contacts,
					struct doca_gpu_buf_arr *buf_arr_gpu_page_not_found, uint32_t nbytes_page_not_found)
{
	doca_error_t ret;
	struct doca_gpu_eth_txq *txq = NULL;
	struct doca_gpu_buf *buf = NULL;
	uintptr_t buf_addr;
	struct eth_ip_tcp_hdr *hdr;
	uint8_t *payload;
	uint32_t base_pkt_len = sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr);
	uint16_t send_pkts = 0;
	uint32_t nbytes_page = 0;
	uint32_t lane_id = threadIdx.x % WARP_SIZE;
	uint32_t warp_id = threadIdx.x / WARP_SIZE;

	/* Use global atomic counter for TX buffer index allocation */
	uint64_t doca_gpu_buf_idx = 0;

	if (warp_id == 0) {
		txq = txq0;
	} else if (warp_id == 1) {
		txq = txq1;
	} else if (warp_id == 2) {
		txq = txq2;
	} else if (warp_id == 3) {
		txq = txq3;
	} else {
		return;
	}

	/* Poll ring_buffer slots for RESULT_READY (CPU processed results) */
	int ring_check_idx = warp_id;

	while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
		send_pkts = 0;
		if (txq && g_inference_ring_buf != nullptr) {
			bool found_ready = false;
			int slot_idx = -1;
			struct inference_ring_slot *current_slot = nullptr;

			/* Poll for RESULT_READY slots */
			for (int poll_offset = 0; poll_offset < INFERENCE_RING_SIZE && !found_ready; poll_offset++) {
				int check_idx = (ring_check_idx + poll_offset) % INFERENCE_RING_SIZE;

				if (g_sem_inference_gpu != nullptr) {
					enum doca_gpu_semaphore_status sem_status;
					doca_gpu_dev_semaphore_get_status(g_sem_inference_gpu, check_idx, &sem_status);
					if (sem_status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
						uint32_t slot_status = g_inference_ring_buf->slots[check_idx].ready;
						if (slot_status == UVM_STATUS_RESULT_READY) {
							slot_idx = check_idx;
							current_slot = &g_inference_ring_buf->slots[slot_idx];
							found_ready = true;
						}
					}
				} else {
					/* Fallback: direct slot status check */
					uint32_t slot_status = g_inference_ring_buf->slots[check_idx].ready;
					if (slot_status == UVM_STATUS_RESULT_READY) {
						slot_idx = check_idx;
						current_slot = &g_inference_ring_buf->slots[slot_idx];
						found_ready = true;
					}
				}
			}

			ring_check_idx = (ring_check_idx + 1) % INFERENCE_RING_SIZE;

			if (!found_ready) {
				continue;
			}

			if (lane_id == 0) {
				printf("[HTTP_SERVER] Found RESULT_READY slot %d (warp %d)\n", slot_idx, warp_id);
			}

			/* Claim slot atomically */
			if (current_slot != nullptr) {
				uint32_t old_val = 0;
				if (lane_id == 0) {
					uint32_t expected = UVM_STATUS_RESULT_READY;
					old_val = atomicCAS((unsigned int*)&current_slot->ready, expected, UVM_STATUS_CONSUMED);
				}
				/* Broadcast result to all lanes */
				old_val = __shfl_sync(0xffffffff, old_val, 0);

				if (old_val != UVM_STATUS_RESULT_READY) {
					continue;  /* Another warp claimed this slot */
				}

				/* Reset semaphore */
				if (g_sem_inference_gpu != nullptr) {
					doca_gpu_dev_semaphore_set_status(g_sem_inference_gpu, slot_idx, DOCA_GPU_SEMAPHORE_STATUS_FREE);
				}

				/* Atomic TX buffer index allocation */
				if (lane_id == 0)
					doca_gpu_buf_idx = atomicAdd(&g_tx_buf_counter, 1ULL) % TX_BUF_NUM;
				doca_gpu_buf_idx = __shfl_sync(0xffffffff, doca_gpu_buf_idx, 0);

				enum http_page_get page_type = (enum http_page_get)current_slot->http_page_type;

				if (page_type == HTTP_GET_INDEX) {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_index, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_index;
				} else if (page_type == HTTP_GET_CONTACTS) {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_contacts, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_contacts;
				} else if (page_type == HTTP_GET_INFERENCE) {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_index, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_index; /* Will be updated with JSON response */
				} else {
					ret = doca_gpu_dev_buf_get_buf(buf_arr_gpu_page_not_found, doca_gpu_buf_idx, &buf);
					if (ret != DOCA_SUCCESS) {
						printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
						DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
						break;
					}
					nbytes_page = nbytes_page_not_found;
				}

				ret = doca_gpu_dev_buf_get_addr(buf, &buf_addr);
				if (ret != DOCA_SUCCESS) {
					printf("Error %d doca_gpu_dev_eth_rxq_get_buf block %d thread %d\n", ret, warp_id, lane_id);
					DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
					break;
				}

				/* Generate JSON response for inference requests */
				if (page_type == HTTP_GET_INFERENCE) {
					char *response_buf = (char *)(buf_addr + base_pkt_len);

					/* Zero-Copy Response Construction */
					int body_len = 0;
					int header_len = 0;
					bool result_ready = false;
					uint64_t request_id = current_slot->request_id;

					/* Result already in current_slot->data */
					if (lane_id == 0 && slot_idx >= 0) {
						result_ready = true;
						char *body_start = response_buf + HTTP_BODY_TEMP_OFFSET;
						body_len = strcpy_to_buf(body_start, current_slot->data, HTTP_BODY_MAX_LEN);
						body_start[body_len] = '\0';
						current_slot->t8_gpu_sent = clock64();
						(void)request_id;
					}

					result_ready = __shfl_sync(0xffffffff, result_ready, 0);

					/* Lane 0 constructs HTTP response */
					if (lane_id == 0) {
						if (!result_ready) {
							const char *error_msg =
								"{\r\n"
								"  \"status\": \"error\",\r\n"
								"  \"message\": \"Ring buffer full or timeout\"\r\n"
								"}\r\n";
							char *body_start = response_buf + HTTP_BODY_TEMP_OFFSET;
							body_len = strcpy_to_buf(body_start, error_msg, HTTP_BODY_MAX_LEN);
							body_start[body_len] = '\0';
						}

						/* Build HTTP headers */
						for (int i = 0; i < HTTP_RESP_PREFIX_LEN; i++)
							response_buf[i] = HTTP_RESP_PREFIX[i];
						header_len = HTTP_RESP_PREFIX_LEN;
						header_len += int_to_str(response_buf + header_len, body_len);
						for (int i = 0; i < HTTP_RESP_SUFFIX_LEN; i++)
							response_buf[header_len + i] = HTTP_RESP_SUFFIX[i];
						header_len += HTTP_RESP_SUFFIX_LEN;
						nbytes_page = header_len + body_len;
					}

					/* Broadcast to all lanes for parallel copy */
					header_len = __shfl_sync(0xffffffff, header_len, 0);
					body_len = __shfl_sync(0xffffffff, body_len, 0);
					nbytes_page = __shfl_sync(0xffffffff, nbytes_page, 0);

					/* Warp-level parallel body copy */
					char *body_src = response_buf + HTTP_BODY_TEMP_OFFSET;
					warp_memcpy(response_buf + header_len, body_src, body_len, lane_id);
				}

				raw_to_tcp(buf_addr, &hdr, &payload);
				/* Read TCP connection info from slot */
				http_set_mac_addr(hdr, (uint16_t *)current_slot->eth_dst_addr_bytes, (uint16_t *)current_slot->eth_src_addr_bytes);
				hdr->l3_hdr.src_addr = current_slot->ip_dst_addr;
				hdr->l3_hdr.dst_addr = current_slot->ip_src_addr;
				hdr->l4_hdr.src_port = current_slot->tcp_dst_port;
				hdr->l4_hdr.dst_port = current_slot->tcp_src_port;
				hdr->l4_hdr.sent_seq = current_slot->tcp_recv_ack;
				/* TCP sequence calculation: client_seq + client_data_len */
				uint32_t client_data_len = BYTE_SWAP16(current_slot->ip_total_length) - sizeof(struct ipv4_hdr) - ((current_slot->tcp_dt_off >> 4) * 4);
				hdr->l4_hdr.recv_ack = BYTE_SWAP32(BYTE_SWAP32(current_slot->tcp_sent_seq) + client_data_len);
				
				/* Keep-Alive: ACK + PSH without FIN */
				hdr->l4_hdr.tcp_flags = TCP_FLAG_ACK | TCP_FLAG_PSH;
				hdr->l4_hdr.cksum = 0;

				hdr->l3_hdr.total_length = BYTE_SWAP16(sizeof(struct ipv4_hdr) + sizeof(struct tcp_hdr) + nbytes_page);
				hdr->l3_hdr.hdr_checksum = 0;  /* Hardware computes checksum */

				ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf, base_pkt_len + nbytes_page, 0);
				if (ret != DOCA_SUCCESS) {
					printf("Error %d doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, lane_id);
					DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
					break;
				}

				/* Reset slot status to FREE */
				__threadfence_system();
				atomicExch((unsigned int*)&current_slot->ready, UVM_STATUS_FREE);

				send_pkts++;
			}
		}
		__syncwarp();

		/* Send only if needed */
#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2)
			send_pkts += __shfl_down_sync(WARP_FULL_MASK, send_pkts, offset);
		__syncwarp();

		if (lane_id == 0 && send_pkts > 0) {
			doca_gpu_dev_eth_txq_commit_strong(txq);
			doca_gpu_dev_eth_txq_push(txq);
		}
		__syncwarp();
	}
}

extern "C" {

doca_error_t kernel_http_server(cudaStream_t stream, uint32_t *exit_cond, struct rxq_tcp_queues *tcp_queues, struct txq_http_queues *http_queues)
{
	cudaError_t result = cudaSuccess;

	if (tcp_queues == NULL || tcp_queues->numq == 0 || exit_cond == 0) {
		DOCA_LOG_ERR("kernel_http_server invalid input values");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/*
	 * Assume no more than MAX_QUEUE (4) receive queues
	 */
	cuda_kernel_http_server<<<1, tcp_queues->numq * WARP_SIZE, 0, stream>>>(exit_cond,
								http_queues->eth_txq_gpu[0], http_queues->eth_txq_gpu[1], http_queues->eth_txq_gpu[2], http_queues->eth_txq_gpu[3],
								tcp_queues->nums,
								tcp_queues->sem_http_gpu[0], tcp_queues->sem_http_gpu[1], tcp_queues->sem_http_gpu[2], tcp_queues->sem_http_gpu[3],
								http_queues->buf_page_index.buf_arr_gpu, http_queues->buf_page_index.pkt_nbytes,
								http_queues->buf_page_contacts.buf_arr_gpu, http_queues->buf_page_contacts.pkt_nbytes,
								http_queues->buf_page_not_found.buf_arr_gpu, http_queues->buf_page_not_found.pkt_nbytes
								);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

/* Set UVM buffer for inference data storage (CPU-side) */
doca_error_t set_inference_ring_buffer_kernel(cudaStream_t stream, struct inference_ring_buffer *ring)
{
	cudaError_t result = cudaSuccess;

	if (ring == NULL) {
		DOCA_LOG_ERR("set_inference_ring_buffer_kernel invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check CUDA state */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Launch setup kernel */
	set_inference_ring_buffer<<<1, 1, 0, stream>>>(ring);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Synchronize to ensure kernel completion */
	result = cudaStreamSynchronize(stream);
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda sync failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

/* Set inference semaphore GPU handle */
doca_error_t set_inference_semaphore_kernel(cudaStream_t stream, struct doca_gpu_semaphore_gpu *sem)
{
	cudaError_t result = cudaSuccess;

	if (sem == NULL) {
		DOCA_LOG_ERR("set_inference_semaphore_kernel invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check CUDA state */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Launch setup kernel */
	set_inference_semaphore<<<1, 1, 0, stream>>>(sem);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Synchronize to ensure kernel completion */
	result = cudaStreamSynchronize(stream);
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda sync failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("Inference semaphore GPU handle set successfully");
	return DOCA_SUCCESS;
}

/* Set request notification semaphore GPU handle */
doca_error_t set_request_semaphore_kernel(cudaStream_t stream, struct doca_gpu_semaphore_gpu *sem)
{
	cudaError_t result = cudaSuccess;

	if (sem == NULL) {
		DOCA_LOG_ERR("set_request_semaphore_kernel invalid input");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check CUDA state */
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Launch setup kernel */
	set_request_semaphore<<<1, 1, 0, stream>>>(sem);
	result = cudaGetLastError();
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	/* Synchronize to ensure kernel completion */
	result = cudaStreamSynchronize(stream);
	if (cudaSuccess != result) {
		DOCA_LOG_ERR("[%s:%d] cuda sync failed with %s \n", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	DOCA_LOG_INFO("Request semaphore GPU handle set successfully");
	return DOCA_SUCCESS;
}

} /* extern C */
