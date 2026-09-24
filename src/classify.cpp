#include "gliner/gliner.h"
#include "internal.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

thread_local char last_error[512] = {};

void set_error(const char * message) {
    std::snprintf(last_error, sizeof(last_error), "%s", message ? message : "unknown error");
}

void clear_error() { last_error[0] = '\0'; }

struct GraphContext {
    ggml_context * value;

    explicit GraphContext(size_t bytes) : value(ggml_init({bytes, nullptr, true})) {
        if (!value) throw std::runtime_error("ggml context allocation failed");
    }
    ~GraphContext() { ggml_free(value); }
    GraphContext(const GraphContext &) = delete;
    GraphContext & operator=(const GraphContext &) = delete;
};

void load_backends_once() {
    static std::once_flag flag;
    std::call_once(flag, [] { ggml_backend_load_all(); });
}

float sigmoid(float x) {
    if (x >= 0) return 1.f / (1.f + std::exp(-x));
    const float value = std::exp(x);
    return value / (1.f + value);
}

} // namespace

struct gliner_context {
    gliner::GgufFile file;
    GraphContext weights;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_tensor * w1 = nullptr;
    ggml_tensor * b1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * b2 = nullptr;
    int hidden = 0;
    int layers = 0;
    int tensors = 0;

    explicit gliner_context(const char * path)
        : file(path), weights(2 * 1024 * 1024) {
        gliner::validate_deberta_metadata(file);
        hidden = file.u32("deberta.hidden_size");
        layers = file.u32("deberta.num_hidden_layers");
        tensors = file.tensor_count();
        load_backends_once();
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend) throw std::runtime_error("No ggml CPU backend available");
        try {
            std::vector<unsigned char> d1, d2, d3, d4;
            w1 = file.read_tensor(weights.value, "classifier.0.weight", d1);
            b1 = file.read_tensor(weights.value, "classifier.0.bias", d2);
            w2 = file.read_tensor(weights.value, "classifier.2.weight", d3);
            b2 = file.read_tensor(weights.value, "classifier.2.bias", d4);
            if (w1->ne[0] != hidden || w1->ne[1] != 2 * hidden ||
                b1->ne[0] != 2 * hidden || w2->ne[0] != 2 * hidden ||
                w2->ne[1] != 1 || b2->ne[0] != 1) {
                throw std::runtime_error("Classifier tensor shapes do not match hidden_size");
            }
            weight_buffer = ggml_backend_alloc_ctx_tensors(weights.value, backend);
            if (!weight_buffer) throw std::runtime_error("Cannot allocate classifier weights");
            ggml_backend_tensor_set(w1, d1.data(), 0, d1.size());
            ggml_backend_tensor_set(b1, d2.data(), 0, d2.size());
            ggml_backend_tensor_set(w2, d3.data(), 0, d3.size());
            ggml_backend_tensor_set(b2, d4.data(), 0, d4.size());
        } catch (...) {
            if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
            ggml_backend_free(backend);
            throw;
        }
    }

    ~gliner_context() {
        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (backend) ggml_backend_free(backend);
    }
};

struct gliner_state {
    const gliner_context * owner;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocator = nullptr;
    std::vector<gliner_score> scores;

    explicit gliner_state(const gliner_context * ctx) : owner(ctx) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend) throw std::runtime_error("No ggml CPU backend available");
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!allocator) {
            ggml_backend_free(backend);
            throw std::runtime_error("Cannot allocate GGML graph allocator");
        }
    }

    ~gliner_state() {
        if (allocator) ggml_gallocr_free(allocator);
        if (backend) ggml_backend_free(backend);
    }
};

extern "C" {

struct gliner_context * gliner_init_from_file(const char * gguf_path) {
    clear_error();
    if (!gguf_path || !*gguf_path) {
        set_error("GGUF path is required");
        return nullptr;
    }
    try {
        return new gliner_context(gguf_path);
    } catch (const std::exception & error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("Unknown model loading error");
        return nullptr;
    }
}

struct gliner_state * gliner_init_state(struct gliner_context * ctx) {
    clear_error();
    if (!ctx) {
        set_error("Context is required");
        return nullptr;
    }
    try {
        return new gliner_state(ctx);
    } catch (const std::exception & error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("Unknown state initialization error");
        return nullptr;
    }
}

void gliner_free(struct gliner_context * ctx) { delete ctx; }
void gliner_free_state(struct gliner_state * state) { delete state; }
const char * gliner_last_error(void) { return last_error; }

int gliner_model_hidden_size(const struct gliner_context * ctx) { return ctx ? ctx->hidden : 0; }
int gliner_model_n_tensors(const struct gliner_context * ctx) { return ctx ? ctx->tensors : 0; }
int gliner_model_n_layers(const struct gliner_context * ctx) { return ctx ? ctx->layers : 0; }

int gliner_score_label_states(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const float * label_states,
    int n_labels,
    int n_threads) {
    clear_error();
    if (state) state->scores.clear();
    if (!ctx || !state || state->owner != ctx || !label_states || n_labels <= 0 || n_threads <= 0) {
        set_error("Invalid context, state, label states, label count, or thread count");
        return GLINER_STATUS_INVALID_ARGUMENT;
    }
    const size_t count = static_cast<size_t>(n_labels);
    const size_t hidden = static_cast<size_t>(ctx->hidden);
    if (count > std::numeric_limits<size_t>::max() / hidden / sizeof(float)) {
        set_error("Label state dimensions overflow");
        return GLINER_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count * hidden; ++i) {
        if (!std::isfinite(label_states[i])) {
            set_error("Label states must be finite");
            return GLINER_STATUS_INVALID_ARGUMENT;
        }
    }
    try {
        GraphContext eval(4 * 1024 * 1024);
        ggml_tensor * input = ggml_new_tensor_2d(eval.value, GGML_TYPE_F32, ctx->hidden, n_labels);
        ggml_set_input(input);
        ggml_tensor * h = ggml_mul_mat(eval.value, ctx->w1, input);
        ggml_tensor * b1 = ctx->b1->type == GGML_TYPE_F32 ? ctx->b1 :
            ggml_cast(eval.value, ctx->b1, GGML_TYPE_F32);
        h = ggml_relu(eval.value, ggml_add(eval.value, h, b1));
        ggml_tensor * out = ggml_mul_mat(eval.value, ctx->w2, h);
        ggml_tensor * b2 = ctx->b2->type == GGML_TYPE_F32 ? ctx->b2 :
            ggml_cast(eval.value, ctx->b2, GGML_TYPE_F32);
        out = ggml_add(eval.value, out, b2);
        ggml_set_output(out);
        ggml_cgraph * graph = ggml_new_graph(eval.value);
        ggml_build_forward_expand(graph, out);
        if (!ggml_gallocr_alloc_graph(state->allocator, graph)) {
            set_error("Cannot allocate classification graph");
            return GLINER_STATUS_BACKEND_ERROR;
        }
        ggml_backend_tensor_set(input, label_states, 0, count * hidden * sizeof(float));
        const auto device = ggml_backend_get_device(state->backend);
        const auto registry = ggml_backend_dev_backend_reg(device);
        const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
        if (set_threads) set_threads(state->backend, n_threads);
        else if (n_threads != 1) {
            set_error("This ggml CPU backend cannot set the thread count");
            return GLINER_STATUS_BACKEND_ERROR;
        }
        if (ggml_backend_graph_compute(state->backend, graph) != GGML_STATUS_SUCCESS) {
            set_error("ggml classification graph failed");
            return GLINER_STATUS_BACKEND_ERROR;
        }
        std::vector<float> logits(count);
        ggml_backend_tensor_get(out, logits.data(), 0, count * sizeof(float));
        state->scores.reserve(count);
        for (float logit : logits) state->scores.push_back({logit, sigmoid(logit)});
        return GLINER_STATUS_OK;
    } catch (const std::exception & error) {
        state->scores.clear();
        set_error(error.what());
        return GLINER_STATUS_MODEL_ERROR;
    } catch (...) {
        state->scores.clear();
        set_error("Unknown classification error");
        return GLINER_STATUS_MODEL_ERROR;
    }
}

int gliner_n_scores(const struct gliner_state * state) {
    return state ? static_cast<int>(state->scores.size()) : 0;
}

const struct gliner_score * gliner_get_scores(const struct gliner_state * state) {
    return state && !state->scores.empty() ? state->scores.data() : nullptr;
}

} // extern "C"
