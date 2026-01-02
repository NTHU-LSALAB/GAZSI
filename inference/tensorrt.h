/*
 * GPU Packet Processing - TensorRT Inference Runner Header
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#ifdef __cplusplus
#include <string>
// Use a forward declaration (void*) to hide TensorRT implementation details
// from other C++ files that include this header. This avoids forcing them
// to include heavy TensorRT headers.
using TensorRT_Model_t = void;
#else
// For C compilation, just use void*
typedef void TensorRT_Model_t;
#endif

/**
 * @brief Initializes pre-allocated GPU buffers for TensorRT inference
 * @param gpu_id CUDA device ID to use
 * @return 0 on success, -1 on failure
 */
#ifdef __cplusplus
extern "C" {
#endif

int init_tensorrt_gpu_buffers(int gpu_id);

/**
 * @brief Loads a TensorRT engine from a file and prepares it for inference.
 * @param engine_path Path to the .engine file.
 * @return A handle to the TensorRT model, or nullptr on failure.
 */
TensorRT_Model_t* load_tensorrt_engine(const char* engine_path);

/**
 * @brief Context Pool inference: uses a specific pre-created execution context
 * @param model A handle to the loaded TensorRT model (with context pool).
 * @param context_id ID of the context to use (0 to NUM_CONTEXTS-1).
 * @param text Input text to process.
 * @param embeddings Output embeddings array.
 * @param token_count Output token count.
 */
void simple_tokenize_and_infer_with_context(TensorRT_Model_t* model, int context_id, const char* text, float* embeddings, int* token_count);

/**
 * @brief Dynamic Batching: batch tokenize and inference for multiple texts
 *
 * Combines N serial inferences into 1 batch inference for efficiency
 *
 * @param model A handle to the loaded TensorRT model.
 * @param texts Array of input text pointers (batch_size elements)
 * @param batch_size Number of texts in the batch (1-4)
 * @param batch_embeddings Output: batch_size x 768 floats (caller allocates)
 * @param token_counts Output: batch_size token counts (caller allocates)
 */
void batch_tokenize_and_infer(TensorRT_Model_t* model, const char** texts, int batch_size,
                               float* batch_embeddings, int* token_counts);

#ifdef __cplusplus
}
#endif
