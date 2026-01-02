/*
 * GPU Packet Processing - TensorRT Inference Runner
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorrt.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <sstream>
#include <cstring>  // for memset, memcpy (pinned memory optimization)
#include "NvInfer.h"
#include <cuda_runtime_api.h>

// Constants for MiniLM model
#define SEQUENCE_LENGTH 128
#define EMBEDDING_DIM 768
#define NUM_CONTEXTS 4

// Context Pool: Pre-allocated GPU buffers + Pinned Host Memory per execution context
struct ContextBuffers {
    // GPU device memory
    void *d_input_ids;
    void *d_attention_mask;
    void *d_token_type_ids;
    void *d_output;
    void *d_pooler_output;  // For secondary output tensor

    // Pinned host memory (for fast DMA transfer)
    int64_t *h_input_ids;        // cudaHostAlloc pinned memory
    int64_t *h_attention_mask;   // cudaHostAlloc pinned memory
    float *h_output;             // cudaHostAlloc pinned memory for output

    // Dedicated CUDA stream for async operations
    cudaStream_t stream;
};

static std::vector<ContextBuffers> g_context_buffers(NUM_CONTEXTS);


extern "C" int init_tensorrt_gpu_buffers(int gpu_id) {
    cudaSetDevice(gpu_id);
    size_t input_size = SEQUENCE_LENGTH * sizeof(int64_t);
    size_t output_size = SEQUENCE_LENGTH * EMBEDDING_DIM * sizeof(float);
    size_t pooler_size = EMBEDDING_DIM * sizeof(float);

    // Allocate GPU buffers + Pinned Host Memory for Context Pool
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        // 1. Allocate GPU device memory
        if (cudaMalloc(&g_context_buffers[i].d_input_ids, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_attention_mask, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_token_type_ids, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_output, output_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_pooler_output, pooler_size) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to allocate GPU buffers for context %d\n", i);
            return -1;
        }

        // 2. Allocate Pinned Host Memory (for fast DMA)
        if (cudaHostAlloc(&g_context_buffers[i].h_input_ids, input_size, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_attention_mask, input_size, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_output, EMBEDDING_DIM * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to allocate pinned host memory for context %d\n", i);
            return -1;
        }

        // 3. Create dedicated CUDA stream
        if (cudaStreamCreateWithFlags(&g_context_buffers[i].stream, cudaStreamNonBlocking) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to create CUDA stream for context %d\n", i);
            return -1;
        }

        // Initialize token_type_ids to 0
        if (cudaMemset(g_context_buffers[i].d_token_type_ids, 0, input_size) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to initialize token_type_ids for context %d\n", i);
            return -1;
        }

        fprintf(stderr, "[CONTEXT_POOL] Context %d: GPU buffers + pinned memory + stream (input=%zu, output=%zu)\n",
                i, input_size, output_size);
    }

    fprintf(stderr, "[TensorRT] GPU buffers pre-allocated (%d contexts)\n", NUM_CONTEXTS);
    return 0;
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// Simple logger for TensorRT messages
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

/*
 * Tokenization Module
 *
 * Simple word-based tokenization for MiniLM model
 * Format: [CLS] word1 word2 ... wordN [SEP] [PAD] [PAD] ...
 */

/*
 * tokenize_text - Convert text to token IDs and attention mask
 *
 * @text:           Input text
 * @input_ids:      Output token IDs (must be pre-allocated SEQUENCE_LENGTH)
 * @attention_mask: Output attention mask (must be pre-allocated SEQUENCE_LENGTH)
 *
 * Returns: Actual token count
 */
static int tokenize_text(const char* text,
                         int64_t* input_ids,
                         int64_t* attention_mask) {
    memset(input_ids, 0, SEQUENCE_LENGTH * sizeof(int64_t));
    memset(attention_mask, 0, SEQUENCE_LENGTH * sizeof(int64_t));

    // [CLS] token at position 0
    input_ids[0] = 101;
    attention_mask[0] = 1;

    if (!text || text[0] == '\0') {
        // Empty text: only [CLS] and [SEP]
        input_ids[1] = 102;
        attention_mask[1] = 1;
        return 2;
    }

    // Simple word-based tokenization
    std::string text_str(text);
    std::istringstream iss(text_str);
    std::string word;
    int token_pos = 1;
    int actual_tokens = 1;

    while (std::getline(iss, word, ' ') && token_pos < SEQUENCE_LENGTH - 1) {
        if (!word.empty()) {
            // Hash-based token ID generation
            uint32_t hash = 0;
            for (char c : word) {
                hash = hash * 31 + (uint8_t)c;
            }
            input_ids[token_pos] = 1000 + (hash % 20000);
            attention_mask[token_pos] = 1;
            token_pos++;
            actual_tokens++;
        }
    }

    // [SEP] token
    if (token_pos < SEQUENCE_LENGTH) {
        input_ids[token_pos] = 102;
        attention_mask[token_pos] = 1;
        actual_tokens++;
    }

    return actual_tokens;
}

// Wrapper struct to hold all TensorRT objects
struct TensorRT_Context {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;  // Legacy single context
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;  // Context Pool

    // Dynamic tensor names (detected from engine)
    std::string output_tensor_name;       // "last_hidden_state" or "output"
    std::string pooler_tensor_name;       // Secondary output tensor (e.g., "1895", "1001")
    bool has_token_type_ids;              // Whether model needs token_type_ids
    bool has_pooler_output;               // Whether model has pooler output
    int embedding_dim;                    // 384 or 768
};

extern "C" TensorRT_Model_t* load_tensorrt_engine(const char* engine_path) {
    Logger logger;
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Could not read engine file: " << engine_path << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_blob(size);
    file.read(engine_blob.data(), size);
    file.close();

    auto model = new TensorRT_Context();
    model->runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!model->runtime) {
        std::cerr << "Error: Failed to create TensorRT Runtime." << std::endl;
        delete model;
        return nullptr;
    }

    model->engine.reset(model->runtime->deserializeCudaEngine(engine_blob.data(), size));
    if (!model->engine) {
        std::cerr << "Error: Failed to deserialize TensorRT Engine." << std::endl;
        delete model;
        return nullptr;
    }

    // Create legacy single context (backward compatibility)
    model->context.reset(model->engine->createExecutionContext());
    if (!model->context) {
        std::cerr << "Error: Failed to create TensorRT Execution Context." << std::endl;
        delete model;
        return nullptr;
    }

    // Detect tensor names from engine
    model->has_token_type_ids = false;
    model->has_pooler_output = false;
    model->output_tensor_name = "output";  // Default for new models
    model->pooler_tensor_name = "";
    model->embedding_dim = 768;            // Default

    int num_tensors = model->engine->getNbIOTensors();
    std::cout << "[TensorRT] Detecting tensors from engine (" << num_tensors << " tensors):" << std::endl;

    for (int i = 0; i < num_tensors; i++) {
        const char* name = model->engine->getIOTensorName(i);
        auto mode = model->engine->getTensorIOMode(name);
        auto shape = model->engine->getTensorShape(name);

        std::cout << "  [" << i << "] " << name << " (";
        std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "): shape=(";
        for (int d = 0; d < shape.nbDims; d++) {
            if (d > 0) std::cout << ", ";
            std::cout << shape.d[d];
        }
        std::cout << ")" << std::endl;

        // Check for token_type_ids
        if (strcmp(name, "token_type_ids") == 0) {
            model->has_token_type_ids = true;
        }

        // Check for output tensor name
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (strcmp(name, "last_hidden_state") == 0) {
                model->output_tensor_name = "last_hidden_state";
            } else if (strcmp(name, "output") == 0) {
                model->output_tensor_name = "output";
            } else {
                // Secondary output (pooler output like "1895", "1001")
                model->has_pooler_output = true;
                model->pooler_tensor_name = name;
            }
            // Get embedding dim from output shape (last dimension)
            if (shape.nbDims >= 3) {
                int dim = shape.d[shape.nbDims - 1];
                if (dim > 0) {  // Not dynamic
                    model->embedding_dim = dim;
                }
            }
        }
    }

    std::cout << "[TensorRT] Model config: output_tensor=" << model->output_tensor_name
              << ", has_token_type_ids=" << (model->has_token_type_ids ? "yes" : "no")
              << ", has_pooler=" << (model->has_pooler_output ? model->pooler_tensor_name : "no")
              << ", embedding_dim=" << model->embedding_dim << std::endl;

    // Create Context Pool
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        auto ctx = model->engine->createExecutionContext();
        if (!ctx) {
            std::cerr << "[CONTEXT_POOL] Error: Failed to create execution context " << i << std::endl;
            delete model;
            return nullptr;
        }
        model->contexts.emplace_back(ctx);
        std::cout << "[CONTEXT_POOL] Created context " << i << std::endl;
    }

    return model;
}


// Context Pool version - use specified context with dedicated buffers
extern "C" void simple_tokenize_and_infer_with_context(TensorRT_Model_t* model_ptr, int context_id,
                                                         const char* text, float* embeddings, int* token_count) {
    auto* model = static_cast<TensorRT_Context*>(model_ptr);

    // Validate parameters
    if (!model || !text || !embeddings || !token_count) {
        fprintf(stderr, "[CONTEXT_POOL] Error: Invalid parameters\n");
        if (token_count) *token_count = 0;
        return;
    }

    if (context_id < 0 || context_id >= NUM_CONTEXTS) {
        fprintf(stderr, "[CONTEXT_POOL] Error: Invalid context_id %d (must be 0-%d)\n", context_id, NUM_CONTEXTS-1);
        *token_count = 0;
        return;
    }

    if (model->contexts.empty() || !model->contexts[context_id]) {
        fprintf(stderr, "[CONTEXT_POOL] Error: Context %d not initialized\n", context_id);
        *token_count = 0;
        return;
    }

    try {
        // Use context pool's dedicated buffers (with pinned memory)
        auto& buffers = g_context_buffers[context_id];
        auto* context = model->contexts[context_id].get();

        // Tokenize directly into pinned memory
        *token_count = tokenize_text(text, buffers.h_input_ids, buffers.h_attention_mask);

        size_t input_size = SEQUENCE_LENGTH * sizeof(int64_t);
        size_t output_size = EMBEDDING_DIM * sizeof(float);

        // Use pre-allocated stream and cudaMemcpyAsync (pinned memory supports DMA)
        cudaStream_t stream = buffers.stream;

        // Async copy to GPU (pinned memory → device memory via DMA)
        CUDA_CHECK(cudaMemcpyAsync(buffers.d_input_ids, buffers.h_input_ids, input_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buffers.d_attention_mask, buffers.h_attention_mask, input_size, cudaMemcpyHostToDevice, stream));

        // Set input shapes
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = 1;  // batch size
        input_dims.d[1] = SEQUENCE_LENGTH;

        context->setInputShape("input_ids", input_dims);
        context->setInputShape("attention_mask", input_dims);
        if (model->has_token_type_ids) {
            context->setInputShape("token_type_ids", input_dims);
        }

        // Bind tensors
        context->setTensorAddress("input_ids", buffers.d_input_ids);
        context->setTensorAddress("attention_mask", buffers.d_attention_mask);
        if (model->has_token_type_ids) {
            context->setTensorAddress("token_type_ids", buffers.d_token_type_ids);
        }
        context->setTensorAddress(model->output_tensor_name.c_str(), buffers.d_output);
        // Bind pooler output if present
        if (model->has_pooler_output && !model->pooler_tensor_name.empty()) {
            context->setTensorAddress(model->pooler_tensor_name.c_str(), buffers.d_pooler_output);
        }

        // Execute inference (async on same stream)
        context->enqueueV3(stream);

        // Async copy result back (device memory → pinned memory via DMA)
        CUDA_CHECK(cudaMemcpyAsync(buffers.h_output, buffers.d_output, output_size, cudaMemcpyDeviceToHost, stream));

        // Wait for all operations to complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy from pinned memory to user's buffer (fast memcpy)
        memcpy(embeddings, buffers.h_output, output_size);

    } catch (const std::exception& e) {
        fprintf(stderr, "[CONTEXT_POOL] Error in context %d: %s\n", context_id, e.what());
        *token_count = 0;
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            embeddings[i] = 0.0f;
        }
    }
}

/*
 * Dynamic Batching Implementation
 *
 * Combines N serial inferences into 1 batch inference
 * Batch of 4: ~5ms vs serial 4 × 4ms = 16ms
 */

#define MAX_BATCH_SIZE 8

// Pre-allocated batch buffers to avoid malloc per inference
struct BatchBuffers {
    // GPU device memory (batch_size × sequence_length)
    void *d_input_ids;
    void *d_attention_mask;
    void *d_token_type_ids;
    void *d_output;
    void *d_pooler_output;

    // Pinned host memory
    int64_t *h_input_ids;
    int64_t *h_attention_mask;
    float *h_output;

    // Dedicated stream
    cudaStream_t stream;

    bool initialized;
};

static BatchBuffers g_batch_buffers = {nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, false};

// Initialize Batch Buffers (called once)
static void init_batch_buffers() {
    if (g_batch_buffers.initialized) return;

    size_t input_size = MAX_BATCH_SIZE * SEQUENCE_LENGTH * sizeof(int64_t);
    size_t output_size = MAX_BATCH_SIZE * SEQUENCE_LENGTH * EMBEDDING_DIM * sizeof(float);
    size_t pooler_size = MAX_BATCH_SIZE * EMBEDDING_DIM * sizeof(float);

    // GPU buffers
    if (cudaMalloc(&g_batch_buffers.d_input_ids, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_attention_mask, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_token_type_ids, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_output, output_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_pooler_output, pooler_size) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Failed to allocate GPU buffers\n");
        return;
    }

    // Initialize token_type_ids to 0
    cudaMemset(g_batch_buffers.d_token_type_ids, 0, input_size);

    // Pinned host memory
    if (cudaHostAlloc(&g_batch_buffers.h_input_ids, input_size, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&g_batch_buffers.h_attention_mask, input_size, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&g_batch_buffers.h_output, output_size, cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Failed to allocate pinned memory\n");
        return;
    }

    // Stream
    if (cudaStreamCreateWithFlags(&g_batch_buffers.stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Failed to create stream\n");
        return;
    }

    g_batch_buffers.initialized = true;
    fprintf(stderr, "[BATCH] Batch buffers initialized (max_batch=%d)\n", MAX_BATCH_SIZE);
}

// Batch Tokenization: Convert N texts to batch tensor
static void batch_tokenize(const char** texts, int batch_size,
                            int64_t* batch_input_ids, int64_t* batch_attention_mask,
                            int* token_counts) {
    for (int b = 0; b < batch_size; b++) {
        int64_t* input_ids = &batch_input_ids[b * SEQUENCE_LENGTH];
        int64_t* attention_mask = &batch_attention_mask[b * SEQUENCE_LENGTH];
        token_counts[b] = tokenize_text(texts[b], input_ids, attention_mask);
    }
}

extern "C" void batch_tokenize_and_infer(TensorRT_Model_t* model_ptr, const char** texts, int batch_size,
                                          float* batch_embeddings, int* token_counts) {
    auto* model = static_cast<TensorRT_Context*>(model_ptr);

    // Validate parameters
    if (!model || !texts || !batch_embeddings || !token_counts) {
        fprintf(stderr, "[BATCH] Error: Invalid parameters\n");
        return;
    }

    if (batch_size < 1 || batch_size > MAX_BATCH_SIZE) {
        fprintf(stderr, "[BATCH] Error: Invalid batch_size %d (must be 1-%d)\n", batch_size, MAX_BATCH_SIZE);
        return;
    }

    // Ensure batch buffers are initialized
    init_batch_buffers();
    if (!g_batch_buffers.initialized) {
        fprintf(stderr, "[BATCH] Error: Batch buffers not initialized\n");
        return;
    }

    // Use context 0 for inference
    if (model->contexts.empty() || !model->contexts[0]) {
        fprintf(stderr, "[BATCH] Error: Context 0 not available\n");
        return;
    }

    try {
        auto* context = model->contexts[0].get();
        cudaStream_t stream = g_batch_buffers.stream;

        // 1. Batch Tokenization → pinned memory
        batch_tokenize(texts, batch_size,
                       g_batch_buffers.h_input_ids, g_batch_buffers.h_attention_mask,
                       token_counts);

        // 2. H2D Copy (async, pinned memory → GPU)
        size_t input_size = batch_size * SEQUENCE_LENGTH * sizeof(int64_t);
        size_t output_size = batch_size * SEQUENCE_LENGTH * EMBEDDING_DIM * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.d_input_ids, g_batch_buffers.h_input_ids,
                                    input_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.d_attention_mask, g_batch_buffers.h_attention_mask,
                                    input_size, cudaMemcpyHostToDevice, stream));

        // 3. Set batch input shapes
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = batch_size;  // BATCH SIZE!
        input_dims.d[1] = SEQUENCE_LENGTH;

        context->setInputShape("input_ids", input_dims);
        context->setInputShape("attention_mask", input_dims);
        if (model->has_token_type_ids) {
            context->setInputShape("token_type_ids", input_dims);
        }

        // 4. Bind tensors
        context->setTensorAddress("input_ids", g_batch_buffers.d_input_ids);
        context->setTensorAddress("attention_mask", g_batch_buffers.d_attention_mask);
        if (model->has_token_type_ids) {
            context->setTensorAddress("token_type_ids", g_batch_buffers.d_token_type_ids);
        }
        context->setTensorAddress(model->output_tensor_name.c_str(), g_batch_buffers.d_output);
        // Bind pooler output if present (required by TensorRT - all outputs must be bound)
        if (model->has_pooler_output && !model->pooler_tensor_name.empty()) {
            context->setTensorAddress(model->pooler_tensor_name.c_str(), g_batch_buffers.d_pooler_output);
        }

        // 5. Execute batch inference
        if (!context->enqueueV3(stream)) {
            fprintf(stderr, "[BATCH] Error: enqueueV3 failed\n");
            return;
        }

        // 6. D2H Copy (async, GPU → pinned memory)
        CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.h_output, g_batch_buffers.d_output,
                                    output_size, cudaMemcpyDeviceToHost, stream));

        // 7. Wait for completion
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 8. Extract [CLS] embeddings for each batch item
        // Output shape: [batch_size, seq_len, embedding_dim]
        // [CLS] token is at position 0 for each batch item
        for (int b = 0; b < batch_size; b++) {
            float* src = &g_batch_buffers.h_output[b * SEQUENCE_LENGTH * EMBEDDING_DIM];  // [CLS] at position 0
            float* dst = &batch_embeddings[b * EMBEDDING_DIM];
            memcpy(dst, src, EMBEDDING_DIM * sizeof(float));
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "[BATCH] Exception: %s\n", e.what());
        for (int b = 0; b < batch_size; b++) {
            token_counts[b] = 0;
            memset(&batch_embeddings[b * EMBEDDING_DIM], 0, EMBEDDING_DIM * sizeof(float));
        }
    }
}
