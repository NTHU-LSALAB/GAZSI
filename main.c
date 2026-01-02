/*
 * GPU Packet Processing - Main Application
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdlib.h>
#include <string.h>
#include <rte_ethdev.h>

#include "common.h"
#include "tcp/session.h"
#include "tcp/cpu_rss.h"
#include "inference/tensorrt.h"
#include "inference/ring_buffer.h"  /* Ring Buffer UVM - supports concurrency */
#include <pthread.h>
#include <ctype.h>

#define SLEEP_IN_NANOS (10 * 1000) /* Sample the PE every 10 microseconds  */

/* Buffer and inference constants */
#define DATA_BUFFER_SIZE     1024    /* Max size for inference data buffer */
#define EMBEDDING_DIM        768     /* Output embedding dimension */
#define MAX_DATA_LEN         1000    /* Maximum valid data length */
#define SPIN_ITERATIONS      500     /* Busy-wait spin iterations */
#define IDLE_SLEEP_US        50      /* Sleep time when no requests (microseconds) */
#define BATCH_WAIT_US        500     /* Initial batch wait time (microseconds) */

/*
 * URL decode function - converts %XX encoding back to original characters
 * Example: %20 -> space, %2B -> +
 */
static void url_decode(char *str) {
    char *src = str;
    char *dst = str;

    while (*src) {
        if (*src == '%' && isxdigit((unsigned char)src[1]) && isxdigit((unsigned char)src[2])) {
            /* Decode %XX */
            char hex[3] = {src[1], src[2], '\0'};
            *dst++ = (char)strtol(hex, NULL, 16);
            src += 3;
        } else if (*src == '+') {
            /* '+' also represents space */
            *dst++ = ' ';
            src++;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
}

DOCA_LOG_REGISTER(GPUNET);

bool force_quit;
static struct doca_gpu *gpu_dev;
static struct app_gpu_cfg app_cfg = {0};
static struct doca_dev *ddev;
static uint16_t dpdk_dev_port_id;
static struct rxq_tcp_queues tcp_queues;
static struct txq_http_queues http_queues;
static struct doca_flow_port *df_port;

/* Simple UVM inference buffer - Linus-style global variable */
static struct inference_ring_buffer *g_inference_ring_buf = NULL;  /* Ring Buffer - supports concurrent requests */

/* Context Pool: concurrent inference threads, each bound to an execution context */
#define NUM_INFERENCE_THREADS 1  /* Single thread + Batching is more efficient: 1 batch inference is faster than 4 separate inferences */
static pthread_t inference_reader_threads[NUM_INFERENCE_THREADS];
static bool inference_reader_running = false;

struct reader_thread_args {
	int thread_id;
	int context_id;
};

static struct doca_pe *pe;

/* UVM inference processing */
static TensorRT_Model_t *tensorrt_model = NULL;
static int g_cuda_id = 0;  /* CUDA device ID */
/* Context Pool removes global mutex - each thread uses independent context for lock-free concurrency */

/*
 * Get CPU timestamp in nanoseconds - for comparison with GPU clock64()
 */
static inline uint64_t get_timestamp_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/*
 * ============================================================================
 * Dynamic Batching Inference Reader
 * ============================================================================
 *
 * Collect multiple PARAM_READY slots and process in one batch inference.
 *
 * Parameters:
 * - BATCH_MAX_SIZE: Maximum batch size supported by TensorRT engine (dynamic batch 1~8)
 * - BATCH_COLLECT_WINDOW_US: Time window to collect concurrent requests
 */
#define BATCH_MAX_SIZE 8               /* Max concurrent batch size */
#define BATCH_COLLECT_WINDOW_US 5000   /* 5ms collection window */

/*
 * Collect PARAM_READY slots (non-blocking, using CAS atomic claim)
 * Returns the number of slots collected
 *
 * Uses memory barriers and boundary checks to handle concurrent GPU writes.
 */
static int collect_ready_slots(struct inference_ring_buffer *ring,
                               uint32_t *batch_slots,
                               char batch_data[][DATA_BUFFER_SIZE],
                               uint32_t *batch_lens,
                               int max_batch) {
	int count = 0;

	for (uint32_t i = 0; i < INFERENCE_RING_SIZE && count < max_batch; i++) {
		struct inference_ring_slot *slot = &ring->slots[i];

		/* Atomic CAS claim: PARAM_READY -> PROCESSING */
		uint32_t expected = UVM_STATUS_PARAM_READY;
		if (__atomic_compare_exchange_n(&slot->ready,
		                                 &expected,
		                                 UVM_STATUS_PROCESSING,
		                                 false,
		                                 __ATOMIC_ACQ_REL,
		                                 __ATOMIC_ACQUIRE)) {
			/* Memory barrier to ensure seeing GPU-written data */
			__sync_synchronize();

			/* Boundary check - slot->len must be within reasonable range */
			uint32_t data_len = slot->len;
			if (data_len > MAX_DATA_LEN) {
				DOCA_LOG_WARN("Slot %u has invalid len=%u, resetting to FREE", i, data_len);
				__atomic_store_n(&slot->ready, UVM_STATUS_FREE, __ATOMIC_RELEASE);
				continue;
			}

			/* Copy data */
			batch_slots[count] = i;
			batch_lens[count] = data_len;

			uint32_t copy_len = (data_len < DATA_BUFFER_SIZE - 1) ? data_len : DATA_BUFFER_SIZE - 1;
			memcpy(batch_data[count], slot->data, copy_len);
			batch_data[count][copy_len] = '\0';

			/* URL decode */
			url_decode(batch_data[count]);

			/* T3: CPU detected data */
			slot->t3_cpu_read = get_timestamp_ns();

			count++;
		}
	}

	return count;
}

/*
 * Inference reader thread with dynamic batching
 */
void* simple_inference_reader(void* arg) {
	struct reader_thread_args *args = (struct reader_thread_args*)arg;
	int thread_id = args->thread_id;
	(void)args->context_id;

	DOCA_LOG_INFO("[BATCH] Inference reader thread %d started (max_batch=%d)",
	              thread_id, BATCH_MAX_SIZE);

	cudaSetDevice(0);
	DOCA_LOG_INFO("[BATCH] Thread %d: Using GPU 0 for TensorRT", thread_id);

	/* Batch buffers */
	uint32_t batch_slots[BATCH_MAX_SIZE];
	char batch_data[BATCH_MAX_SIZE][DATA_BUFFER_SIZE];
	uint32_t batch_lens[BATCH_MAX_SIZE];
	const char *batch_texts[BATCH_MAX_SIZE];
	float batch_embeddings[BATCH_MAX_SIZE * EMBEDDING_DIM];
	int batch_token_counts[BATCH_MAX_SIZE];
	char result_buffer[DATA_BUFFER_SIZE];

	struct timespec start_time, end_time;

	while (inference_reader_running && !force_quit) {
		if (g_inference_ring_buf == NULL) {
			usleep(100);
			continue;
		}

		/* Step 1: Collect ready slots */
		int batch_size = collect_ready_slots(g_inference_ring_buf, batch_slots,
		                                      batch_data, batch_lens, BATCH_MAX_SIZE);

		if (batch_size == 0) {
			/* No requests, busy-wait briefly */
			for (int spin = 0; spin < SPIN_ITERATIONS; spin++) {
				__asm__ __volatile__("pause" ::: "memory");
			}
			usleep(IDLE_SLEEP_US);
			continue;
		}

		/* Step 2: Wait for more concurrent requests if applicable */
		if (batch_size == 1) {
			usleep(BATCH_WAIT_US);
			int additional = collect_ready_slots(g_inference_ring_buf,
			                                      &batch_slots[batch_size],
			                                      &batch_data[batch_size],
			                                      &batch_lens[batch_size],
			                                      BATCH_MAX_SIZE - batch_size);
			if (additional > 0) {
				batch_size += additional;
				DOCA_LOG_DBG("[BATCH] Found %d more requests, batch_size=%d", additional, batch_size);
				if (batch_size < BATCH_MAX_SIZE) {
					usleep(BATCH_COLLECT_WINDOW_US - BATCH_WAIT_US);
					batch_size += collect_ready_slots(g_inference_ring_buf,
					                                   &batch_slots[batch_size],
					                                   &batch_data[batch_size],
					                                   &batch_lens[batch_size],
					                                   BATCH_MAX_SIZE - batch_size);
				}
			}
		} else if (batch_size > 1 && batch_size < BATCH_MAX_SIZE) {
			usleep(BATCH_COLLECT_WINDOW_US);
			batch_size += collect_ready_slots(g_inference_ring_buf,
			                                   &batch_slots[batch_size],
			                                   &batch_data[batch_size],
			                                   &batch_lens[batch_size],
			                                   BATCH_MAX_SIZE - batch_size);
		}

		/* Step 3: Prepare batch texts */
		for (int i = 0; i < batch_size; i++) {
			batch_texts[i] = batch_data[i];
		}

		DOCA_LOG_INFO("[BATCH] Collected %d requests, starting batch inference", batch_size);

		/* Step 4: Batch Inference */
		if (tensorrt_model != NULL) {
			clock_gettime(CLOCK_MONOTONIC, &start_time);

			/* T4: TensorRT inference start */
			uint64_t t4 = get_timestamp_ns();
			for (int i = 0; i < batch_size; i++) {
				g_inference_ring_buf->slots[batch_slots[i]].t4_tensorrt_start = t4;
			}

			batch_tokenize_and_infer(tensorrt_model, batch_texts, batch_size,
			                          batch_embeddings, batch_token_counts);

			/* T5: TensorRT inference end */
			uint64_t t5 = get_timestamp_ns();
			for (int i = 0; i < batch_size; i++) {
				g_inference_ring_buf->slots[batch_slots[i]].t5_tensorrt_end = t5;
			}

			clock_gettime(CLOCK_MONOTONIC, &end_time);
			long elapsed_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
			                  (end_time.tv_nsec - start_time.tv_nsec) / 1000;

			DOCA_LOG_INFO("[BATCH] Batch inference complete: batch_size=%d, time=%ldμs (%.1fμs/req)",
			              batch_size, elapsed_us, (float)elapsed_us / batch_size);

			/* Step 5: Distribute results to slots */
			for (int i = 0; i < batch_size; i++) {
				uint32_t slot_idx = batch_slots[i];
				float *emb = &batch_embeddings[i * EMBEDDING_DIM];
				int tokens = batch_token_counts[i];

				int result_len = snprintf(result_buffer, sizeof(result_buffer),
					"{\"input\":\"%s\",\"tokens\":%d,\"embedding_sample\":[%.6f,%.6f,%.6f],\"inference_time_us\":%ld,\"batch_size\":%d}",
					batch_data[i], tokens,
					emb[0], emb[1], emb[2],
					elapsed_us, batch_size);

				/* T6: CPU wrote result */
				g_inference_ring_buf->slots[slot_idx].t6_cpu_wrote_uvm = get_timestamp_ns();

				if (cpu_write_inference_result_to_gpu_ring(g_inference_ring_buf, slot_idx, result_buffer)) {
					DOCA_LOG_INFO("[BATCH] Wrote result to slot %u (len=%d)", slot_idx, result_len);
				} else {
					DOCA_LOG_ERR("[BATCH] Failed to write result to slot %u", slot_idx);
				}
			}
		} else {
			/* Fallback: TensorRT not loaded */
			DOCA_LOG_WARN("[BATCH] TensorRT not loaded, using fallback");
			for (int i = 0; i < batch_size; i++) {
				uint32_t slot_idx = batch_slots[i];
				snprintf(result_buffer, sizeof(result_buffer),
					"{\"input\":\"%.900s\",\"status\":\"no_model\"}", batch_data[i]);
				cpu_write_inference_result_to_gpu_ring(g_inference_ring_buf, slot_idx, result_buffer);
			}
		}
	}

	DOCA_LOG_INFO("[BATCH] Inference reader thread %d stopped", thread_id);
	return NULL;
}

/*
 * DOCA PE callback to be invoked if any Eth Txq get an error
 * sending packets.
 *
 * @event_error [in]: DOCA PE event error handler
 * @event_user_data [in]: custom user data set at registration time
 */
void error_send_packet_cb(struct doca_eth_txq_gpu_event_error_send_packet *event_error, union doca_data event_user_data)
{
	uint16_t packet_index;

	doca_eth_txq_gpu_event_error_send_packet_get_position(event_error, &packet_index);
	DOCA_LOG_INFO("Error in send queue %ld, packet %d. Gracefully killing the app",
		      event_user_data.u64,
		      packet_index);
	DOCA_GPUNETIO_VOLATILE(force_quit) = true;
}

/*
 * Get timestamp in nanoseconds
 *
 * @sec [out]: seconds
 * @return: UTC timestamp
 */
static uint64_t get_ns(uint64_t *sec)
{
	struct timespec t;
	int ret;

	ret = clock_gettime(CLOCK_REALTIME, &t);
	if (ret != 0)
		exit(EXIT_FAILURE);

	(*sec) = (uint64_t)t.tv_sec;

	return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

/*
 * CPU thread to print statistics from GPU filtering on the console
 *
 * @args [in]: thread input args
 */
static void stats_core(void *args)
{
	(void)args;

	doca_error_t result = DOCA_SUCCESS;
	enum doca_gpu_semaphore_status status;
	struct stats_tcp tcp_st[MAX_QUEUES] = {0};
	uint32_t sem_idx_tcp[MAX_QUEUES] = {0};
	uint64_t start_time_sec = 0;
	uint64_t interval_print = 0;
	uint64_t interval_sec = 0;
	struct stats_tcp *custom_tcp_st;

	DOCA_LOG_INFO("Core %u is reporting filter stats", rte_lcore_id());
	get_ns(&start_time_sec);
	interval_print = get_ns(&interval_sec);
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		/* Check TCP packets */
		for (int idxq = 0; idxq < tcp_queues.numq; idxq++) {
			result = doca_gpu_semaphore_get_status(tcp_queues.sem_cpu[idxq], sem_idx_tcp[idxq], &status);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("TCP semaphore error");
				DOCA_GPUNETIO_VOLATILE(force_quit) = true;
				return;
			}

			if (status == DOCA_GPU_SEMAPHORE_STATUS_READY) {
				result = doca_gpu_semaphore_get_custom_info_addr(tcp_queues.sem_cpu[idxq],
										 sem_idx_tcp[idxq],
										 (void **)&(custom_tcp_st));
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("TCP semaphore get address error");
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				tcp_st[idxq].http += custom_tcp_st->http;
				tcp_st[idxq].http_head += custom_tcp_st->http_head;
				tcp_st[idxq].http_get += custom_tcp_st->http_get;
				tcp_st[idxq].http_post += custom_tcp_st->http_post;
				tcp_st[idxq].tcp_syn += custom_tcp_st->tcp_syn;
				tcp_st[idxq].tcp_fin += custom_tcp_st->tcp_fin;
				tcp_st[idxq].tcp_ack += custom_tcp_st->tcp_ack;
				tcp_st[idxq].others += custom_tcp_st->others;
				tcp_st[idxq].total += custom_tcp_st->total;

				result = doca_gpu_semaphore_set_status(tcp_queues.sem_cpu[idxq],
								       sem_idx_tcp[idxq],
								       DOCA_GPU_SEMAPHORE_STATUS_FREE);
				if (result != DOCA_SUCCESS) {
					DOCA_LOG_ERR("TCP semaphore %d error", sem_idx_tcp[idxq]);
					DOCA_GPUNETIO_VOLATILE(force_quit) = true;
					return;
				}

				sem_idx_tcp[idxq] = (sem_idx_tcp[idxq] + 1) % tcp_queues.nums;
			}
		}

		if ((get_ns(&interval_sec) - interval_print) > 5000000000) {
			printf("\nSeconds %ld\n", interval_sec - start_time_sec);

			for (int idxq = 0; idxq < tcp_queues.numq; idxq++) {
				printf("[TCP] QUEUE: %d HTTP: %d GET: %d POST: %d SYN: %d FIN: %d ACK: %d OTHER: %d TOTAL: %d\n",
				       idxq,
				       tcp_st[idxq].http,
				       tcp_st[idxq].http_get,
				       tcp_st[idxq].http_post,
				       tcp_st[idxq].tcp_syn,
				       tcp_st[idxq].tcp_fin,
				       tcp_st[idxq].tcp_ack,
				       tcp_st[idxq].others,
				       tcp_st[idxq].total);
			}

			interval_print = get_ns(&interval_sec);
		}
	}
}

/*
 * Signal handler to quit application gracefully
 *
 * @signum [in]: signal received
 */
static void signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		DOCA_LOG_INFO("Signal %d received, preparing to exit!", signum);
		DOCA_GPUNETIO_VOLATILE(force_quit) = true;
	}
}

/*
 * GPU packet processing application main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int main(int argc, char **argv)
{
	doca_error_t result;
	int current_lcore = 0;
	int cuda_id;
	cudaError_t cuda_ret;
	struct doca_log_backend *sdk_log;
	struct timespec ts = {
		.tv_sec = 0,
		.tv_nsec = SLEEP_IN_NANOS,
	};

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	DOCA_LOG_INFO("===========================================================");
	DOCA_LOG_INFO("DOCA version: %s", doca_version());
	DOCA_LOG_INFO("===========================================================");

	/* Basic DPDK initialization */
	result = doca_argp_init("gpunet", &app_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = register_application_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	DOCA_LOG_INFO("Options enabled:\n\tGPU %s\n\tNIC %s\n\tGPU Rx queues %d\n\tGPU HTTP server enabled %s",
		      app_cfg.gpu_pcie_addr,
		      app_cfg.nic_pcie_addr,
		      app_cfg.queue_num,
		      (app_cfg.http_server == true ? "Yes" : "No"));

	/* In a multi-GPU system, ensure CUDA refers to the right GPU device */
	cuda_ret = cudaDeviceGetByPCIBusId(&cuda_id, app_cfg.gpu_pcie_addr);
	if (cuda_ret != cudaSuccess) {
		DOCA_LOG_ERR("Invalid GPU bus id provided %s", app_cfg.gpu_pcie_addr);
		return DOCA_ERROR_INVALID_VALUE;
	}

	g_cuda_id = cuda_id;  /* Save to global for other threads */
	cudaFree(0);
	cudaSetDevice(cuda_id);

	result = init_doca_device(app_cfg.nic_pcie_addr, &ddev, &dpdk_dev_port_id);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function init_doca_device returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Initialize DOCA GPU instance */
	result = doca_gpu_create(app_cfg.gpu_pcie_addr, &gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_gpu_create returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Initialize TensorRT on GPU 0 (separate from DOCA kernels on GPU 1) */
	cudaSetDevice(0);
	DOCA_LOG_INFO("Using GPU 0 for TensorRT model");
	if (init_tensorrt_gpu_buffers(0) != 0) {
		DOCA_LOG_ERR("Failed to initialize TensorRT GPU buffers");
		return EXIT_FAILURE;
	}


	/* Use custom engine path if provided, otherwise use default */
	const char *engine_path = (app_cfg.engine_path[0] != '\0') ?
		app_cfg.engine_path :
		"/opt/mellanox/doca/applications/gpu_packet_processing_v2/models/minilm.engine";
	DOCA_LOG_INFO("Loading TensorRT engine: %s", engine_path);
	tensorrt_model = load_tensorrt_engine(engine_path);
	if (tensorrt_model) {
		DOCA_LOG_INFO("TensorRT model loaded (GPU 0)");

		/* Warmup TensorRT to eliminate cold start latency */
		DOCA_LOG_INFO("[WARMUP] Warming up TensorRT contexts...");
		float temp_embeddings[1024];
		int temp_token_count;
		#define WARMUP_ITERATIONS 10

		const char *warmup_long_text =
			"The quick brown fox jumps over the lazy dog. "
			"This is a comprehensive warmup text designed to exercise "
			"the full tokenization and inference pipeline. "
			"Machine learning models require proper initialization "
			"to achieve optimal performance. GPU memory access patterns "
			"and CUDA kernel execution paths should be exercised "
			"before serving real user requests. This ensures consistent "
			"low latency for all subsequent inference operations.";

		const char *warmup_short_text = "hello world test";

		for (int i = 0; i < NUM_INFERENCE_THREADS; i++) {
			DOCA_LOG_INFO("[WARMUP] Warming up context %d...", i);

			for (int j = 0; j < 5; j++) {
				simple_tokenize_and_infer_with_context(tensorrt_model, i,
					warmup_long_text, temp_embeddings, &temp_token_count);
			}

			for (int j = 0; j < 5; j++) {
				simple_tokenize_and_infer_with_context(tensorrt_model, i,
					warmup_short_text, temp_embeddings, &temp_token_count);
			}

			DOCA_LOG_INFO("[WARMUP] Context %d done (%d iterations)", i, WARMUP_ITERATIONS);
		}
		DOCA_LOG_INFO("[WARMUP] All %d contexts warmed up (%d total inferences)", NUM_INFERENCE_THREADS, NUM_INFERENCE_THREADS * WARMUP_ITERATIONS);
	} else {
		DOCA_LOG_WARN("TensorRT model loading failed, inference will be skipped");
	}

	/* Switch back to DOCA GPU */
	cudaSetDevice(cuda_id);
	DOCA_LOG_INFO("Switching back to GPU %d for DOCA initialization", cuda_id);


	df_port = init_doca_flow(dpdk_dev_port_id, app_cfg.queue_num);
	if (df_port == NULL) {
		DOCA_LOG_ERR("FAILED: init_doca_flow");
		return EXIT_FAILURE;
	}

	result = doca_pe_create(&pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create pe queue: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = create_tcp_queues(&tcp_queues,
				   df_port,
				   gpu_dev,
				   ddev,
				   app_cfg.queue_num,
				   app_cfg.queue_num,  /* sem_num equals queue_num, each warp handles one semaphore */
				   app_cfg.http_server,
				   &http_queues,
				   pe,
				   &error_send_packet_cb);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_tcp_queues returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Create root control pipe to route tcp packets */
	result = create_root_pipe(&tcp_queues, df_port);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function create_root_pipe returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	/* Gracefully terminate app if ctrlc */
	DOCA_GPUNETIO_VOLATILE(force_quit) = false;
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	cudaStream_t rx_tcp_stream, tx_http_server;
	cudaError_t res_rt = cudaSuccess;
	uint32_t *cpu_exit_condition;
	uint32_t *gpu_exit_condition;

	res_rt = cudaStreamCreateWithFlags(&rx_tcp_stream, cudaStreamNonBlocking);
	if (res_rt != cudaSuccess) {
		DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
		return EXIT_FAILURE;
	}

	if (app_cfg.http_server) {
		res_rt = cudaStreamCreateWithFlags(&tx_http_server, cudaStreamNonBlocking);
		if (res_rt != cudaSuccess) {
			DOCA_LOG_ERR("Function cudaStreamCreateWithFlags error %d", res_rt);
			return EXIT_FAILURE;
		}
	}

	result = doca_gpu_mem_alloc(gpu_dev,
				    sizeof(uint32_t),
				    4096,
				    DOCA_GPU_MEM_TYPE_GPU_CPU,
				    (void **)&gpu_exit_condition,
				    (void **)&cpu_exit_condition);
	if (result != DOCA_SUCCESS || gpu_exit_condition == NULL || cpu_exit_condition == NULL) {
		DOCA_LOG_ERR("Function doca_gpu_mem_alloc returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}
	cpu_exit_condition[0] = 0;

	/*
	 * Some GPUs may require an initial warmup without doing any real operation.
	 */
	DOCA_LOG_INFO("Warm up CUDA kernels");
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
	kernel_receive_tcp(rx_tcp_stream, gpu_exit_condition, &tcp_queues, app_cfg.http_server);
	if (app_cfg.http_server)
		kernel_http_server(tx_http_server, gpu_exit_condition, &tcp_queues, &http_queues);

	cudaStreamSynchronize(rx_tcp_stream);
	if (app_cfg.http_server)
		cudaStreamSynchronize(tx_http_server);
	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 0;

	/* Initialize UVM inference buffer */
	if (app_cfg.http_server) {
		DOCA_LOG_INFO("Initializing UVM inference buffer");
		g_inference_ring_buf = init_inference_ring_buffer(cuda_id);
		if (g_inference_ring_buf == NULL) {
			DOCA_LOG_ERR("Failed to initialize UVM inference buffer");
			return EXIT_FAILURE;
		}

		/* Set GPU-side UVM buffer pointer */
		result = set_inference_ring_buffer_kernel(tx_http_server, g_inference_ring_buf);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to set UVM buffer in GPU: %s", doca_error_get_descr(result));
			return EXIT_FAILURE;
		}

		/* Set CPU-side inference semaphore handle for GPU notification */
		if (tcp_queues.sem_inference_cpu) {
			set_inference_semaphore_cpu(tcp_queues.sem_inference_cpu);
			DOCA_LOG_INFO("Inference semaphore passed to CPU write function");
		}

		/* Set GPU-side inference semaphore handle (GPU waits for CPU notification) */
		if (tcp_queues.sem_inference_gpu) {
			result = set_inference_semaphore_kernel(tx_http_server, tcp_queues.sem_inference_gpu);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to set inference semaphore in GPU: %s",
				             doca_error_get_descr(result));
				return EXIT_FAILURE;
			}
			DOCA_LOG_INFO("Inference semaphore GPU handle set successfully");
		}

		/* Set GPU-side request notification semaphore handle */
		if (tcp_queues.sem_request_gpu) {
			result = set_request_semaphore_kernel(tx_http_server, tcp_queues.sem_request_gpu);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("Failed to set request semaphore in GPU: %s",
				             doca_error_get_descr(result));
				return EXIT_FAILURE;
			}
			DOCA_LOG_INFO("Request semaphore GPU handle set successfully");

			set_request_semaphore_cpu(tcp_queues.sem_request_cpu);
		}

		/* Start CPU-side inference reader threads */
		inference_reader_running = true;
		for (int i = 0; i < NUM_INFERENCE_THREADS; i++) {
			struct reader_thread_args *args = malloc(sizeof(struct reader_thread_args));
			if (!args) {
				DOCA_LOG_ERR("Failed to allocate thread args for thread %d", i);
				inference_reader_running = false;
				return EXIT_FAILURE;
			}
			args->thread_id = i;
			args->context_id = i;

			if (pthread_create(&inference_reader_threads[i], NULL, simple_inference_reader, args) != 0) {
				DOCA_LOG_ERR("Failed to create inference reader thread %d", i);
				free(args);
				inference_reader_running = false;
				return EXIT_FAILURE;
			}
			DOCA_LOG_INFO("[CONTEXT_POOL] Started inference thread %d (context %d)", i, i);
		}
		DOCA_LOG_INFO("[CONTEXT_POOL] All %d inference threads started", NUM_INFERENCE_THREADS);
	}

	DOCA_LOG_INFO("Launching CUDA kernels");

	kernel_receive_tcp(rx_tcp_stream, gpu_exit_condition, &tcp_queues, app_cfg.http_server);
	if (app_cfg.http_server)
		kernel_http_server(tx_http_server, gpu_exit_condition, &tcp_queues, &http_queues);

	/* Launch stats proxy thread to report pipeline status */
	current_lcore = rte_get_next_lcore(current_lcore, true, false);
	if (rte_eal_remote_launch((void *)stats_core, NULL, current_lcore) != 0) {
		DOCA_LOG_ERR("Remote launch failed");
		goto exit;
	}

	if (app_cfg.http_server) {
		tcp_queues.tcp_ack_pkt_pool = rte_pktmbuf_pool_create("tcp_ack_pkt_pool",
								      1023,
								      0,
								      0,
								      RTE_MBUF_DEFAULT_BUF_SIZE,
								      rte_socket_id());
		if (!tcp_queues.tcp_ack_pkt_pool) {
			DOCA_LOG_ERR("%s: failed to allocate tcp-ack packet pool", __func__);
			goto exit;
		}

		/* Start the CPU RSS threads to address new TCP connections */
		tcp_queues.lcore_idx_start = rte_get_next_lcore(current_lcore, true, false);
		for (int i = 0; i < tcp_queues.numq_cpu_rss; i++) {
			current_lcore = rte_get_next_lcore(current_lcore, true, false);
			if (rte_eal_remote_launch(tcp_cpu_rss_func, &tcp_queues, current_lcore) != 0) {
				DOCA_LOG_ERR("Remote launch failed");
				goto exit;
			}
		}
	}

	DOCA_LOG_INFO("Waiting for termination");

	/* Network layer warmup - fork child process to send warmup requests
	 * Environment variables:
	 * - DOCA_WARMUP_NS: Network namespace (default: test_ns)
	 * - DOCA_WARMUP_IP: Server IP (default: 10.0.0.6)
	 * - DOCA_WARMUP_PORT: Server port (default: 8089)
	 * - DOCA_NO_WARMUP: Set to 1 to skip network warmup
	 */
	if (app_cfg.http_server) {
		const char *no_warmup = getenv("DOCA_NO_WARMUP");
		if (no_warmup == NULL || strcmp(no_warmup, "1") != 0) {
			pid_t warmup_pid = fork();
			if (warmup_pid == 0) {
				/* Child process: send warmup requests */
				const char *net_ns = getenv("DOCA_WARMUP_NS");
				const char *server_ip = getenv("DOCA_WARMUP_IP");
				const char *server_port = getenv("DOCA_WARMUP_PORT");

				if (net_ns == NULL) net_ns = "test_ns";
				if (server_ip == NULL) server_ip = "10.0.0.6";
				if (server_port == NULL) server_port = "8089";

				usleep(500000);  /* Wait for parent to be ready */

				char cmd[512];
				DOCA_LOG_INFO("[WARMUP] Starting network layer warmup (5 requests)...");

				for (int i = 0; i < 5; i++) {
					snprintf(cmd, sizeof(cmd),
						"ip netns exec %s curl -s --max-time 3 "
						"'http://%s:%s/inference?d=warmup_%d' > /dev/null 2>&1",
						net_ns, server_ip, server_port, i);
					system(cmd);
					usleep(200000);
				}

				DOCA_LOG_INFO("[WARMUP] Network layer warmup complete!");
				_exit(0);
			} else if (warmup_pid > 0) {
				/* Parent: continue main loop without waiting */
				DOCA_LOG_INFO("[WARMUP] Background network warmup started (PID: %d)", warmup_pid);
			} else {
				DOCA_LOG_WARN("[WARMUP] fork() failed, skipping network warmup");
			}
		} else {
			DOCA_LOG_INFO("[WARMUP] DOCA_NO_WARMUP=1, skipping network warmup");
		}
	}

	/* This loop keeps busy main thread until force_quit is set to 1 (e.g. typing ctrl+c) */
	while (DOCA_GPUNETIO_VOLATILE(force_quit) == false) {
		doca_pe_progress(pe);
		nanosleep(&ts, &ts);
	}

	DOCA_GPUNETIO_VOLATILE(*cpu_exit_condition) = 1;
	cudaStreamSynchronize(rx_tcp_stream);
	cudaStreamDestroy(rx_tcp_stream);
	if (app_cfg.http_server) {
		cudaStreamSynchronize(tx_http_server);
		cudaStreamDestroy(tx_http_server);
	}

	doca_gpu_mem_free(gpu_dev, gpu_exit_condition);

	DOCA_LOG_INFO("GPU work ended");

	current_lcore = 0;
	RTE_LCORE_FOREACH_WORKER(current_lcore)
	{
		if (rte_eal_wait_lcore(current_lcore) < 0) {
			DOCA_LOG_ERR("Bad exit for coreid: %d", current_lcore);
			break;
		}
	}

exit:

	result = destroy_flow_queue(df_port, &tcp_queues, app_cfg.http_server, &http_queues);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function finalize_doca_flow returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_gpu_destroy(gpu_dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy GPU: %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	result = doca_pe_destroy(pe);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Function doca_pe_destroy returned %s", doca_error_get_descr(result));
		return EXIT_FAILURE;
	}

	doca_dev_close(ddev);

	/* Stop all inference reader threads - Context Pool */
	if (inference_reader_running) {
		DOCA_LOG_INFO("[CONTEXT_POOL] Stopping all inference threads...");
		inference_reader_running = false;
		for (int i = 0; i < NUM_INFERENCE_THREADS; i++) {
			pthread_join(inference_reader_threads[i], NULL);
			DOCA_LOG_INFO("[CONTEXT_POOL] Thread %d stopped", i);
		}
		DOCA_LOG_INFO("[CONTEXT_POOL] All inference threads stopped");
	}

	/* Clean up inference ring buffer */
	if (g_inference_ring_buf) {
		free_inference_ring_buffer(g_inference_ring_buf);
		g_inference_ring_buf = NULL;
		DOCA_LOG_INFO("Inference ring buffer cleaned up");
	}

	if (tensorrt_model) {
		tensorrt_model = NULL;
	}

	DOCA_LOG_INFO("Application finished successfully");

	return EXIT_SUCCESS;
}
