/*
 * GPU Packet Processing - TensorRT Inference Runner
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorrt.h"
#include "ring_buffer.h"
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

// Buffer allocation uses MAX_OUTPUT_DIM (vocab_size for decoders); runtime uses model->embedding_dim
#define SEQUENCE_LENGTH 128
#define MAX_OUTPUT_DIM 32768
#define NUM_CONTEXTS 4

// Context Pool: Pre-allocated GPU buffers + Pinned Host Memory per execution context
struct ContextBuffers {
    // GPU device memory
    void *d_input_ids;
    void *d_attention_mask;
    void *d_token_type_ids;
    void *d_output;
    void *d_pooler_output;
    void *d_position_ids;

    // Pinned host memory (for fast DMA transfer)
    int64_t *h_input_ids;
    int64_t *h_attention_mask;
    float *h_output;
    int64_t *h_position_ids;

    // Dedicated CUDA stream for async operations
    cudaStream_t stream;
};

static std::vector<ContextBuffers> g_context_buffers(NUM_CONTEXTS);


extern "C" int init_tensorrt_gpu_buffers(int gpu_id) {
    cudaSetDevice(gpu_id);
    size_t input_size = SEQUENCE_LENGTH * sizeof(int64_t);
    size_t output_size = SEQUENCE_LENGTH * MAX_OUTPUT_DIM * sizeof(float);
    size_t pooler_size = MAX_OUTPUT_DIM * sizeof(float);

    // Allocate GPU buffers + Pinned Host Memory for Context Pool
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        // 1. Allocate GPU device memory
        if (cudaMalloc(&g_context_buffers[i].d_input_ids, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_attention_mask, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_token_type_ids, input_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_output, output_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_pooler_output, pooler_size) != cudaSuccess ||
            cudaMalloc(&g_context_buffers[i].d_position_ids, sizeof(int64_t)) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to allocate GPU buffers for context %d\n", i);
            return -1;
        }

        // 2. Allocate Pinned Host Memory (for fast DMA)
        if (cudaHostAlloc(&g_context_buffers[i].h_input_ids, input_size, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_attention_mask, input_size, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_output, MAX_OUTPUT_DIM * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&g_context_buffers[i].h_position_ids, sizeof(int64_t), cudaHostAllocDefault) != cudaSuccess) {
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

    input_ids[0] = 101;
    attention_mask[0] = 1;

    if (!text || text[0] == '\0') {
        input_ids[1] = 102;
        attention_mask[1] = 1;
        return 2;
    }

    int token_pos = 1;
    uint32_t hash = 0;
    bool in_word = false;

    for (const char *p = text; *p && token_pos < SEQUENCE_LENGTH - 1; p++) {
        if (*p == ' ') {
            if (in_word) {
                input_ids[token_pos] = 1000 + (hash % 20000);
                attention_mask[token_pos] = 1;
                token_pos++;
                hash = 0;
                in_word = false;
            }
        } else {
            hash = hash * 31 + (uint8_t)*p;
            in_word = true;
        }
    }

    if (in_word && token_pos < SEQUENCE_LENGTH - 1) {
        input_ids[token_pos] = 1000 + (hash % 20000);
        attention_mask[token_pos] = 1;
        token_pos++;
    }

    input_ids[token_pos] = 102;
    attention_mask[token_pos] = 1;
    token_pos++;

    return token_pos;
}

// Wrapper struct to hold all TensorRT objects
struct TensorRT_Context {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;  // Legacy single context
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;  // Context Pool

    // Dynamic tensor names (detected from engine)
    std::string output_tensor_name;
    std::string pooler_tensor_name;
    std::string position_ids_name;
    bool has_token_type_ids;
    bool has_pooler_output;
    bool has_position_ids;
    bool output_has_seq_dim;
    int embedding_dim;
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
    model->has_position_ids = false;
    model->output_has_seq_dim = true;
    model->output_tensor_name = "output";
    model->pooler_tensor_name = "";
    model->position_ids_name = "";
    model->embedding_dim = 768;

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

        // Check for position_ids (LLaMA rotary embedding: scalar or 1D input)
        if (mode == nvinfer1::TensorIOMode::kINPUT &&
            strcmp(name, "input_ids") != 0 &&
            strcmp(name, "attention_mask") != 0 &&
            strcmp(name, "token_type_ids") != 0) {
            model->has_position_ids = true;
            model->position_ids_name = name;
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
            // Get output dim from shape (last dimension)
            if (shape.nbDims >= 2) {
                int dim = shape.d[shape.nbDims - 1];
                if (dim > 0) {
                    model->embedding_dim = dim;
                }
            }
            // Encoder: (batch, seq, dim) → 3D; Decoder: (batch, vocab) → 2D
            model->output_has_seq_dim = (shape.nbDims >= 3);
        }
    }

    std::cout << "[TensorRT] Model config: output_tensor=" << model->output_tensor_name
              << ", has_token_type_ids=" << (model->has_token_type_ids ? "yes" : "no")
              << ", has_pooler=" << (model->has_pooler_output ? model->pooler_tensor_name : "no")
              << ", has_position_ids=" << (model->has_position_ids ? model->position_ids_name : "no")
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
        int embed_dim = model->embedding_dim;
        size_t output_size = embed_dim * sizeof(float);

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
        if (model->has_position_ids) {
            buffers.h_position_ids[0] = (int64_t)SEQUENCE_LENGTH;
            CUDA_CHECK(cudaMemcpyAsync(buffers.d_position_ids, buffers.h_position_ids,
                                        sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            context->setTensorAddress(model->position_ids_name.c_str(), buffers.d_position_ids);
        }
        context->setTensorAddress(model->output_tensor_name.c_str(), buffers.d_output);
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
        for (int i = 0; i < model->embedding_dim; i++) {
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
    void *d_position_ids;

    // Pinned host memory
    int64_t *h_input_ids;
    int64_t *h_attention_mask;
    float *h_output;
    int64_t *h_position_ids;

    // Dedicated stream
    cudaStream_t stream;

    bool initialized;

    // CUDA Graph: pre-captured inference graphs per batch size
    // graph_exec[bs] captures H2D + enqueueV3 + D2H for batch_size=bs
    cudaGraphExec_t graph_exec[MAX_BATCH_SIZE + 1];
    bool graphs_ready;
};

static BatchBuffers g_batch_buffers = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr, nullptr, false,
                                        {}, false};

// Initialize Batch Buffers (called once)
static void init_batch_buffers() {
    if (g_batch_buffers.initialized) return;

    size_t input_size = MAX_BATCH_SIZE * SEQUENCE_LENGTH * sizeof(int64_t);
    size_t output_size = MAX_BATCH_SIZE * SEQUENCE_LENGTH * MAX_OUTPUT_DIM * sizeof(float);
    size_t pooler_size = MAX_BATCH_SIZE * MAX_OUTPUT_DIM * sizeof(float);

    // GPU buffers
    if (cudaMalloc(&g_batch_buffers.d_input_ids, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_attention_mask, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_token_type_ids, input_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_output, output_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_pooler_output, pooler_size) != cudaSuccess ||
        cudaMalloc(&g_batch_buffers.d_position_ids, sizeof(int64_t)) != cudaSuccess) {
        fprintf(stderr, "[BATCH] Failed to allocate GPU buffers\n");
        return;
    }

    // Initialize token_type_ids to 0
    cudaMemset(g_batch_buffers.d_token_type_ids, 0, input_size);

    // Pinned host memory
    if (cudaHostAlloc(&g_batch_buffers.h_input_ids, input_size, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&g_batch_buffers.h_attention_mask, input_size, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&g_batch_buffers.h_output, output_size, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&g_batch_buffers.h_position_ids, sizeof(int64_t), cudaHostAllocDefault) != cudaSuccess) {
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

/*
 * CUDA Graph Capture for Batch Inference
 *
 * Pre-captures the TensorRT inference pipeline (H2D + enqueueV3 + D2H) as
 * CUDA Graphs, one per batch size (1 to MAX_BATCH_SIZE). This eliminates
 * per-inference CPU overhead from setInputShape, setTensorAddress, and the
 * ~50-100 internal kernel launches inside enqueueV3, replacing them with a
 * single cudaGraphLaunch call (~4 µs).
 *
 * Prerequisite: init_batch_buffers() must be called first.
 * Buffer addresses are fixed at init time and baked into each graph.
 */
extern "C" int init_cuda_graphs(TensorRT_Model_t* model_ptr) {
    auto* model = static_cast<TensorRT_Context*>(model_ptr);

    if (!model || model->contexts.empty() || !model->contexts[0]) {
        fprintf(stderr, "[CUDA_GRAPH] Error: model or context not ready\n");
        return -1;
    }

    // Ensure batch buffers exist
    init_batch_buffers();
    if (!g_batch_buffers.initialized) {
        fprintf(stderr, "[CUDA_GRAPH] Error: batch buffers not initialized\n");
        return -1;
    }

    auto* context = model->contexts[0].get();
    cudaStream_t stream = g_batch_buffers.stream;

    fprintf(stderr, "[CUDA_GRAPH] Capturing inference graphs for batch sizes 1-%d...\n", MAX_BATCH_SIZE);

    // Determine max batch from engine profile
    auto max_shape = model->engine->getProfileShape("input_ids", 0, nvinfer1::OptProfileSelector::kMAX);
    int engine_max_batch = (max_shape.nbDims >= 1 && max_shape.d[0] > 0) ? max_shape.d[0] : 1;
    int graph_max_batch = std::min(MAX_BATCH_SIZE, engine_max_batch);

    fprintf(stderr, "[CUDA_GRAPH] Engine max batch=%d, capturing graphs for bs 1-%d\n",
            engine_max_batch, graph_max_batch);

    for (int bs = 1; bs <= graph_max_batch; bs++) {
        // 1. Set shapes and addresses OUTSIDE capture (these are CPU-side state)
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = bs;
        input_dims.d[1] = SEQUENCE_LENGTH;

        context->setInputShape("input_ids", input_dims);
        context->setInputShape("attention_mask", input_dims);
        if (model->has_token_type_ids) {
            context->setInputShape("token_type_ids", input_dims);
        }

        context->setTensorAddress("input_ids", g_batch_buffers.d_input_ids);
        context->setTensorAddress("attention_mask", g_batch_buffers.d_attention_mask);
        if (model->has_token_type_ids) {
            context->setTensorAddress("token_type_ids", g_batch_buffers.d_token_type_ids);
        }
        if (model->has_position_ids) {
            g_batch_buffers.h_position_ids[0] = (int64_t)SEQUENCE_LENGTH;
            cudaMemcpy(g_batch_buffers.d_position_ids, g_batch_buffers.h_position_ids,
                       sizeof(int64_t), cudaMemcpyHostToDevice);
            context->setTensorAddress(model->position_ids_name.c_str(), g_batch_buffers.d_position_ids);
        }
        context->setTensorAddress(model->output_tensor_name.c_str(), g_batch_buffers.d_output);
        if (model->has_pooler_output && !model->pooler_tensor_name.empty()) {
            context->setTensorAddress(model->pooler_tensor_name.c_str(), g_batch_buffers.d_pooler_output);
        }

        // 2. Warm-up run: let TensorRT initialize internal cuBLAS handles etc.
        if (!context->enqueueV3(stream)) {
            fprintf(stderr, "[CUDA_GRAPH] Error: warm-up enqueueV3 failed for bs=%d\n", bs);
            return -1;
        }
        cudaStreamSynchronize(stream);

        size_t input_size = bs * SEQUENCE_LENGTH * sizeof(int64_t);
        size_t output_size = model->output_has_seq_dim
            ? bs * SEQUENCE_LENGTH * model->embedding_dim * sizeof(float)
            : bs * model->embedding_dim * sizeof(float);

        cudaGraph_t graph;
        cudaError_t err;

        err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA_GRAPH] Error: cudaStreamBeginCapture failed for bs=%d: %s\n",
                    bs, cudaGetErrorString(err));
            return -1;
        }

        // These memcpy + enqueue operations are recorded, not executed
        cudaMemcpyAsync(g_batch_buffers.d_input_ids, g_batch_buffers.h_input_ids,
                        input_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(g_batch_buffers.d_attention_mask, g_batch_buffers.h_attention_mask,
                        input_size, cudaMemcpyHostToDevice, stream);
        context->enqueueV3(stream);
        cudaMemcpyAsync(g_batch_buffers.h_output, g_batch_buffers.d_output,
                        output_size, cudaMemcpyDeviceToHost, stream);

        err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA_GRAPH] Error: cudaStreamEndCapture failed for bs=%d: %s\n",
                    bs, cudaGetErrorString(err));
            return -1;
        }

        // 4. Instantiate executable graph
        err = cudaGraphInstantiate(&g_batch_buffers.graph_exec[bs], graph, NULL, NULL, 0);
        cudaGraphDestroy(graph);  // Original graph no longer needed
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA_GRAPH] Error: cudaGraphInstantiate failed for bs=%d: %s\n",
                    bs, cudaGetErrorString(err));
            return -1;
        }

        fprintf(stderr, "[CUDA_GRAPH] Captured graph for batch_size=%d (input=%zu, output=%zu bytes)\n",
                bs, input_size, output_size);
    }

    g_batch_buffers.graphs_ready = true;
    fprintf(stderr, "[CUDA_GRAPH] All %d graphs captured successfully\n", graph_max_batch);
    return 0;
}

/*
 * GPU Tokenization Kernel — identical hash-based tokenizer running on GPU.
 * Reads text from ring buffer slots (cudaHostAllocMapped, GPU reads via PCIe).
 * Writes token IDs directly to TRT device buffers (no H2D copy needed).
 *
 * One block per batch item, single thread per block (text parsing is sequential).
 */
__global__ void gpu_tokenize_batch_kernel(
    const char *d_texts,
    int text_stride,
    int64_t *d_input_ids,
    int64_t *d_attention_mask,
    int *d_token_counts,
    int batch_size,
    int seq_len)
{
    int b = blockIdx.x;
    if (b >= batch_size) return;
    if (threadIdx.x != 0) return;

    int64_t *input_ids = &d_input_ids[b * seq_len];
    int64_t *attention_mask = &d_attention_mask[b * seq_len];
    const char *text = &d_texts[b * text_stride];

    for (int i = 0; i < seq_len; i++) {
        input_ids[i] = 0;
        attention_mask[i] = 0;
    }

    input_ids[0] = 101;
    attention_mask[0] = 1;

    int token_pos = 1;
    uint32_t hash = 0;
    bool in_word = false;

    for (int i = 0; i < text_stride - 1 && text[i] != '\0' && token_pos < seq_len - 1; i++) {
        if (text[i] == ' ') {
            if (in_word) {
                input_ids[token_pos] = 1000 + (hash % 20000);
                attention_mask[token_pos] = 1;
                token_pos++;
                hash = 0;
                in_word = false;
            }
        } else {
            hash = hash * 31 + (uint8_t)text[i];
            in_word = true;
        }
    }

    if (in_word && token_pos < seq_len - 1) {
        input_ids[token_pos] = 1000 + (hash % 20000);
        attention_mask[token_pos] = 1;
        token_pos++;
    }

    input_ids[token_pos] = 102;
    attention_mask[token_pos] = 1;
    token_pos++;

    d_token_counts[b] = token_pos;
}

static int *d_token_counts_gpu = nullptr;
static char *d_text_staging = nullptr;
static int64_t *d_gpu_tok_input_ids = nullptr;
static int64_t *d_gpu_tok_attention_mask = nullptr;
static cudaStream_t gpu_tok_stream = nullptr;
#define TEXT_SLOT_SIZE 856

// Batch Tokenization: Convert N texts to batch tensor (CPU path, kept as fallback)
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

        // 1. Batch Tokenization → pinned memory (CPU work, not in graph)
        batch_tokenize(texts, batch_size,
                       g_batch_buffers.h_input_ids, g_batch_buffers.h_attention_mask,
                       token_counts);

        static cudaEvent_t ev_infer_start = nullptr, ev_infer_end = nullptr;
        static double sum_gpu_infer_ms = 0;
        static uint64_t gpu_infer_count = 0;
        if (!ev_infer_start) {
            cudaEventCreate(&ev_infer_start);
            cudaEventCreate(&ev_infer_end);
        }

        if (g_batch_buffers.graphs_ready && batch_size >= 1 && batch_size <= MAX_BATCH_SIZE) {
            cudaEventRecord(ev_infer_start, stream);
            CUDA_CHECK(cudaGraphLaunch(g_batch_buffers.graph_exec[batch_size], stream));
            cudaEventRecord(ev_infer_end, stream);
        } else {
            /* Legacy path: fallback if graphs not captured */
            size_t input_size = batch_size * SEQUENCE_LENGTH * sizeof(int64_t);
            size_t output_size = model->output_has_seq_dim
                ? batch_size * SEQUENCE_LENGTH * model->embedding_dim * sizeof(float)
                : batch_size * model->embedding_dim * sizeof(float);

            CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.d_input_ids, g_batch_buffers.h_input_ids,
                                        input_size, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.d_attention_mask, g_batch_buffers.h_attention_mask,
                                        input_size, cudaMemcpyHostToDevice, stream));

            nvinfer1::Dims input_dims;
            input_dims.nbDims = 2;
            input_dims.d[0] = batch_size;
            input_dims.d[1] = SEQUENCE_LENGTH;

            context->setInputShape("input_ids", input_dims);
            context->setInputShape("attention_mask", input_dims);
            if (model->has_token_type_ids) {
                context->setInputShape("token_type_ids", input_dims);
            }

        context->setTensorAddress("input_ids", d_gpu_tok_input_ids);
        context->setTensorAddress("attention_mask", d_gpu_tok_attention_mask);
            if (model->has_token_type_ids) {
                context->setTensorAddress("token_type_ids", g_batch_buffers.d_token_type_ids);
            }
            if (model->has_position_ids) {
                g_batch_buffers.h_position_ids[0] = (int64_t)SEQUENCE_LENGTH;
                CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.d_position_ids, g_batch_buffers.h_position_ids,
                                            sizeof(int64_t), cudaMemcpyHostToDevice, stream));
                context->setTensorAddress(model->position_ids_name.c_str(), g_batch_buffers.d_position_ids);
            }
            context->setTensorAddress(model->output_tensor_name.c_str(), g_batch_buffers.d_output);
            if (model->has_pooler_output && !model->pooler_tensor_name.empty()) {
                context->setTensorAddress(model->pooler_tensor_name.c_str(), g_batch_buffers.d_pooler_output);
            }

            if (!context->enqueueV3(stream)) {
                fprintf(stderr, "[BATCH] Error: enqueueV3 failed\n");
                return;
            }

            CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.h_output, g_batch_buffers.d_output,
                                        output_size, cudaMemcpyDeviceToHost, stream));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));

        {
            float gpu_ms = 0;
            cudaEventElapsedTime(&gpu_ms, ev_infer_start, ev_infer_end);
            sum_gpu_infer_ms += gpu_ms;
            gpu_infer_count++;
            if (gpu_infer_count > 0 && gpu_infer_count % 2000 == 0)
                fprintf(stderr, "[GPU_INFER] n=%lu avg=%.3fms/batch batch_items=%d\n",
                    (unsigned long)gpu_infer_count, sum_gpu_infer_ms / gpu_infer_count, batch_size);
        }

        int edim = model->embedding_dim;
        int stride = model->output_has_seq_dim ? SEQUENCE_LENGTH * edim : edim;
        for (int b = 0; b < batch_size; b++) {
            float* src = &g_batch_buffers.h_output[b * stride];
            float* dst = &batch_embeddings[b * edim];
            memcpy(dst, src, edim * sizeof(float));
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "[BATCH] Exception: %s\n", e.what());
        int edim = model->embedding_dim;
        for (int b = 0; b < batch_size; b++) {
            token_counts[b] = 0;
            memset(&batch_embeddings[b * edim], 0, edim * sizeof(float));
        }
    }
}

/*
 * GPU-tokenized batch inference: tokenization runs on GPU, reading directly
 * from ring buffer slots. Eliminates CPU tokenization + H2D copy.
 *
 * Data path: ring slot (host-mapped) → GPU tokenize kernel → d_input_ids (GPU)
 *            → TRT inference → d_output (GPU) → D2H → host output
 */
extern "C" void batch_gpu_tokenize_and_infer(
    TensorRT_Model_t* model_ptr,
    struct inference_ring_buffer *ring,
    uint32_t *batch_slots,
    int batch_size,
    float* batch_embeddings,
    int* token_counts)
{
    auto* model = static_cast<TensorRT_Context*>(model_ptr);

    if (!model || !ring || !batch_slots || !batch_embeddings || !token_counts)
        return;
    if (batch_size < 1 || batch_size > MAX_BATCH_SIZE)
        return;

    init_batch_buffers();
    if (!g_batch_buffers.initialized) return;

    /* Use context 1 to avoid state conflicts with CUDA Graphs (captured on context 0) */
    if (model->contexts.size() < 2 || !model->contexts[1]) return;

    if (!d_token_counts_gpu)
        cudaMalloc(&d_token_counts_gpu, MAX_BATCH_SIZE * sizeof(int));
    if (!d_text_staging)
        cudaMalloc(&d_text_staging, MAX_BATCH_SIZE * TEXT_SLOT_SIZE);
    size_t tok_buf_size = MAX_BATCH_SIZE * SEQUENCE_LENGTH * sizeof(int64_t);
    if (!d_gpu_tok_input_ids) {
        cudaMalloc(&d_gpu_tok_input_ids, tok_buf_size);
        cudaMalloc(&d_gpu_tok_attention_mask, tok_buf_size);
        cudaMemset(d_gpu_tok_input_ids, 0, tok_buf_size);
        cudaMemset(d_gpu_tok_attention_mask, 0, tok_buf_size);
    }

    try {
        auto* context = model->contexts[1].get();
        if (!gpu_tok_stream)
            cudaStreamCreate(&gpu_tok_stream);
        cudaStream_t stream = gpu_tok_stream;

        for (int i = 0; i < batch_size; i++) {
            const char *host_data = get_slot_data_host(batch_slots[i]);
            uint32_t len = get_slot_len_host(batch_slots[i]);
            if (len >= TEXT_SLOT_SIZE) len = TEXT_SLOT_SIZE - 1;
            fprintf(stderr, "[GPU_TOK] slot=%u len=%u data=%.20s\n",
                    batch_slots[i], len, host_data ? host_data : "(null)");
            CUDA_CHECK(cudaMemcpyAsync(
                d_text_staging + i * TEXT_SLOT_SIZE,
                host_data, len + 1,
                cudaMemcpyHostToDevice, stream));
        }

        /* Test if ANY GPU operation completes on this stream */
        CUDA_CHECK(cudaMemsetAsync(d_text_staging, 0, 1, stream));
        cudaError_t test_err = cudaStreamSynchronize(stream);
        fprintf(stderr, "[GPU_TOK] memset test: %s\n",
                test_err == cudaSuccess ? "OK" : cudaGetErrorString(test_err));

        fprintf(stderr, "[GPU_TOK] H2D done, testing trivial kernel...\n");

        /* Trivial kernel test: does ANY kernel complete? */
        cudaMemsetAsync(d_gpu_tok_input_ids, 0, 8, stream);
        cudaError_t trivial_err = cudaStreamSynchronize(stream);
        fprintf(stderr, "[GPU_TOK] trivial memset+sync: %s\n",
                trivial_err == cudaSuccess ? "OK" : cudaGetErrorString(trivial_err));

        fprintf(stderr, "[GPU_TOK] launching tokenize kernel bs=%d...\n", batch_size);
        gpu_tokenize_batch_kernel<<<batch_size, 1, 0, stream>>>(
            d_text_staging, TEXT_SLOT_SIZE,
            d_gpu_tok_input_ids,
            d_gpu_tok_attention_mask,
            d_token_counts_gpu,
            batch_size, SEQUENCE_LENGTH);

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            fprintf(stderr, "[GPU_TOK] Kernel launch failed: %s\n",
                    cudaGetErrorString(launch_err));
            return;
        }

        fprintf(stderr, "[GPU_TOK] kernel launched, syncing...\n");
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
            fprintf(stderr, "[GPU_TOK] sync failed: %s\n", cudaGetErrorString(sync_err));
        else
            fprintf(stderr, "[GPU_TOK] sync done!\n");

        /* TRT inference — no H2D copy needed, data already on GPU */
        size_t output_size = model->output_has_seq_dim
            ? batch_size * SEQUENCE_LENGTH * model->embedding_dim * sizeof(float)
            : batch_size * model->embedding_dim * sizeof(float);

        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = SEQUENCE_LENGTH;

        context->setInputShape("input_ids", input_dims);
        context->setInputShape("attention_mask", input_dims);
        if (model->has_token_type_ids)
            context->setInputShape("token_type_ids", input_dims);

        context->setTensorAddress("input_ids", g_batch_buffers.d_input_ids);
        context->setTensorAddress("attention_mask", g_batch_buffers.d_attention_mask);
        if (model->has_token_type_ids)
            context->setTensorAddress("token_type_ids", g_batch_buffers.d_token_type_ids);
        if (model->has_position_ids) {
            g_batch_buffers.h_position_ids[0] = (int64_t)SEQUENCE_LENGTH;
            CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.d_position_ids,
                                        g_batch_buffers.h_position_ids,
                                        sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            context->setTensorAddress(model->position_ids_name.c_str(),
                                       g_batch_buffers.d_position_ids);
        }
        context->setTensorAddress(model->output_tensor_name.c_str(), g_batch_buffers.d_output);
        if (model->has_pooler_output && !model->pooler_tensor_name.empty())
            context->setTensorAddress(model->pooler_tensor_name.c_str(),
                                       g_batch_buffers.d_pooler_output);

        if (!context->enqueueV3(stream)) {
            fprintf(stderr, "[GPU_TOK] Error: enqueueV3 failed\n");
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(g_batch_buffers.h_output, g_batch_buffers.d_output,
                                    output_size, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        /* Copy token counts from GPU after sync (non-pinned dest requires sync first) */
        CUDA_CHECK(cudaMemcpy(token_counts, d_token_counts_gpu,
                               batch_size * sizeof(int), cudaMemcpyDeviceToHost));

        int edim = model->embedding_dim;
        int stride = model->output_has_seq_dim ? SEQUENCE_LENGTH * edim : edim;
        for (int b = 0; b < batch_size; b++) {
            float* src = &g_batch_buffers.h_output[b * stride];
            float* dst = &batch_embeddings[b * edim];
            memcpy(dst, src, edim * sizeof(float));
        }

    } catch (const std::exception& e) {
        fprintf(stderr, "[GPU_TOK] Exception: %s\n", e.what());
        int edim = model->embedding_dim;
        for (int b = 0; b < batch_size; b++) {
            token_counts[b] = 0;
            memset(&batch_embeddings[b * edim], 0, edim * sizeof(float));
        }
    }
}
