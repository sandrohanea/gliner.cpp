#include "gliner/gliner.h"
#include "internal.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <algorithm>
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
    std::unique_ptr<gliner::Tokenizer> tokenizer;
    std::unique_ptr<gliner::Deberta> encoder;

    explicit gliner_context(const char * path)
        : file(path), weights(ggml_tensor_overhead() * static_cast<size_t>(file.tensor_count()) + 1024 * 1024) {
        gliner::validate_deberta_metadata(file);
        hidden = file.u32("deberta.hidden_size");
        layers = file.u32("deberta.num_hidden_layers");
        tensors = file.tensor_count();
        load_backends_once();
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend) throw std::runtime_error("No ggml CPU backend available");
        try {
            if (hidden > 16384) throw std::runtime_error("Unsupported hidden_size");
            w1 = file.tensor(weights.value, "classifier.0.weight", {hidden, 2 * hidden});
            b1 = file.tensor(weights.value, "classifier.0.bias", {2 * hidden});
            w2 = file.tensor(weights.value, "classifier.2.weight", {2 * hidden, 1});
            b2 = file.tensor(weights.value, "classifier.2.bias", {1});
            if (file.has("gliner.format_version")) {
                encoder = std::make_unique<gliner::Deberta>(file, weights.value);
                tokenizer = std::make_unique<gliner::Tokenizer>(file);
            }
            weight_buffer = ggml_backend_alloc_ctx_tensors(weights.value, backend);
            if (!weight_buffer) throw std::runtime_error("Cannot allocate model weights");
            file.load_weights(weights.value);
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
    gliner::Tokenized tokens;
    std::vector<float> label_states;
    gliner_eval_callback callback = nullptr;
    void * callback_data = nullptr;

    void clear() {
        scores.clear();
        tokens.ids.clear();
        tokens.markers.clear();
        label_states.clear();
    }

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

namespace {

ggml_tensor * classification_graph(ggml_context * eval, const gliner_context * ctx, ggml_tensor * input) {
    auto * h = ggml_mul_mat(eval, ctx->w1, input);
    auto * b1 = ctx->b1->type == GGML_TYPE_F32 ? ctx->b1 : ggml_cast(eval, ctx->b1, GGML_TYPE_F32);
    h = ggml_relu(eval, ggml_add(eval, h, b1));
    auto * out = ggml_mul_mat(eval, ctx->w2, h);
    auto * b2 = ctx->b2->type == GGML_TYPE_F32 ? ctx->b2 : ggml_cast(eval, ctx->b2, GGML_TYPE_F32);
    out = ggml_add(eval, out, b2);
    ggml_set_output(out);
    return out;
}

void collect_scores(gliner_state * state, ggml_tensor * out, size_t count) {
    std::vector<float> logits(count);
    ggml_backend_tensor_get(out, logits.data(), 0, count * sizeof(float));
    for (float logit : logits) {
        if (!std::isfinite(logit)) throw std::runtime_error("Model produced nonfinite logits");
        state->scores.push_back({logit, sigmoid(logit)});
    }
}

bool compute(gliner_state * state, ggml_cgraph * graph, int n_threads) {
    const auto device = ggml_backend_get_device(state->backend);
    const auto registry = ggml_backend_dev_backend_reg(device);
    const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
    if (set_threads) set_threads(state->backend, n_threads);
    else if (n_threads != 1) {
        set_error("This ggml CPU backend cannot set the thread count");
        return false;
    }
    if (ggml_backend_graph_compute(state->backend, graph) != GGML_STATUS_SUCCESS) {
        set_error("ggml inference graph failed");
        return false;
    }
    return true;
}

} // namespace

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
int gliner_model_supports_text(const struct gliner_context * ctx) { return ctx && ctx->encoder ? 1 : 0; }

int gliner_score_label_states(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const float * label_states,
    int n_labels,
    int n_threads) {
    clear_error();
    if (state) state->clear();
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
        ggml_tensor * out = classification_graph(eval.value, ctx, input);
        ggml_cgraph * graph = ggml_new_graph(eval.value);
        ggml_build_forward_expand(graph, out);
        if (!ggml_gallocr_alloc_graph(state->allocator, graph)) {
            set_error("Cannot allocate classification graph");
            return GLINER_STATUS_BACKEND_ERROR;
        }
        ggml_backend_tensor_set(input, label_states, 0, count * hidden * sizeof(float));
        if (!compute(state, graph, n_threads)) return GLINER_STATUS_BACKEND_ERROR;
        collect_scores(state, out, count);
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

struct gliner_text_params gliner_default_text_params(void) {
    return {1, 512, 0, nullptr, nullptr};
}

void gliner_set_eval_callback(struct gliner_state * state, gliner_eval_callback callback, void * user_data) {
    if (state) { state->callback = callback; state->callback_data = user_data; }
}

int gliner_classify_text(const struct gliner_context * ctx, struct gliner_state * state,
                        const char * text, const char * task, const char * const * labels,
                        int n_labels, const struct gliner_text_params * params) {
    clear_error();
    if (state) state->clear();
    const auto options = params ? *params : gliner_default_text_params();
    if (!ctx || !state || state->owner != ctx || !text || !task || !labels || n_labels <= 0 ||
        options.n_threads <= 0 || options.max_words < 0 || options.max_tokens <= 0 ||
        options.max_tokens > 4096 || n_labels > options.max_tokens) {
        set_error("Invalid text classification arguments (max_tokens must be 1..4096)");
        return GLINER_STATUS_INVALID_ARGUMENT;
    }
    if (!ctx->encoder || !ctx->tokenizer) {
        set_error("This legacy GGUF supports only label states; reconvert the checkpoint for text inference");
        return GLINER_STATUS_MODEL_ERROR;
    }
    try {
        std::vector<std::string> names;
        std::vector<std::optional<std::string>> descriptions;
        for (int i = 0; i < n_labels; ++i) {
            if (!labels[i]) throw std::invalid_argument("NULL label");
            names.emplace_back(labels[i]);
            if (options.label_descriptions && options.label_descriptions[i]) descriptions.emplace_back(options.label_descriptions[i]);
            else descriptions.emplace_back(std::nullopt);
        }
        auto tokens = ctx->tokenizer->encode(text, task, names, options.prompt ? options.prompt : "",
                                             descriptions, options.max_words, options.max_tokens);
        const int n_tokens = static_cast<int>(tokens.ids.size());
        std::vector<int32_t> c2p, p2c;
        ctx->encoder->relative_indices(n_tokens, c2p, p2c);
        const size_t capacity = static_cast<size_t>(ctx->layers + 1) * 160 + 256;
        GraphContext eval(ggml_tensor_overhead() * capacity * 2 + ggml_graph_overhead_custom(capacity, false));
        auto * ids = ggml_new_tensor_1d(eval.value, GGML_TYPE_I32, n_tokens);
        auto * markers = ggml_new_tensor_1d(eval.value, GGML_TYPE_I32, n_labels);
        auto * c2p_indices = ggml_new_tensor_1d(eval.value, GGML_TYPE_I32, static_cast<int64_t>(c2p.size()));
        auto * p2c_indices = ggml_new_tensor_1d(eval.value, GGML_TYPE_I32, static_cast<int64_t>(p2c.size()));
        for (auto * input : {ids, markers, c2p_indices, p2c_indices}) ggml_set_input(input);
        std::vector<ggml_tensor *> hidden;
        auto * encoded = ctx->encoder->build(eval.value, ids, c2p_indices, p2c_indices, hidden);
        auto * label_states = ggml_get_rows(eval.value, encoded, markers);
        ggml_set_output(label_states);
        if (state->callback) for (auto * layer : hidden) ggml_set_output(layer);
        auto * out = classification_graph(eval.value, ctx, label_states);
        auto * graph = ggml_new_graph_custom(eval.value, capacity, false);
        ggml_build_forward_expand(graph, out);
        if (!ggml_gallocr_alloc_graph(state->allocator, graph)) {
            set_error("Cannot allocate encoder graph");
            return GLINER_STATUS_BACKEND_ERROR;
        }
        ggml_backend_tensor_set(ids, tokens.ids.data(), 0, tokens.ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(markers, tokens.markers.data(), 0, tokens.markers.size() * sizeof(int32_t));
        ggml_backend_tensor_set(c2p_indices, c2p.data(), 0, c2p.size() * sizeof(int32_t));
        ggml_backend_tensor_set(p2c_indices, p2c.data(), 0, p2c.size() * sizeof(int32_t));
        if (!compute(state, graph, options.n_threads)) return GLINER_STATUS_BACKEND_ERROR;
        collect_scores(state, out, static_cast<size_t>(n_labels));
        state->label_states.resize(static_cast<size_t>(n_labels) * ctx->hidden);
        ggml_backend_tensor_get(label_states, state->label_states.data(), 0, state->label_states.size() * sizeof(float));
        if (state->callback) {
            std::vector<float> values(static_cast<size_t>(n_tokens) * ctx->hidden);
            for (size_t i = 0; i < hidden.size(); ++i) {
                ggml_backend_tensor_get(hidden[i], values.data(), 0, values.size() * sizeof(float));
                state->callback(static_cast<int>(i), values.data(), n_tokens, ctx->hidden, state->callback_data);
            }
        }
        state->tokens = std::move(tokens);
        return GLINER_STATUS_OK;
    } catch (const std::invalid_argument & error) {
        state->clear();
        set_error(error.what());
        return GLINER_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception & error) {
        state->clear();
        set_error(error.what());
        return GLINER_STATUS_MODEL_ERROR;
    } catch (...) {
        state->clear();
        set_error("Unknown text classification error");
        return GLINER_STATUS_MODEL_ERROR;
    }
}

int gliner_n_tokens(const struct gliner_state * state) {
    return state ? static_cast<int>(state->tokens.ids.size()) : 0;
}
const int32_t * gliner_get_token_ids(const struct gliner_state * state) {
    return state && !state->tokens.ids.empty() ? state->tokens.ids.data() : nullptr;
}
const int32_t * gliner_get_label_positions(const struct gliner_state * state) {
    return state && !state->tokens.markers.empty() ? state->tokens.markers.data() : nullptr;
}
const float * gliner_get_label_states(const struct gliner_state * state) {
    return state && !state->label_states.empty() ? state->label_states.data() : nullptr;
}

int gliner_n_scores(const struct gliner_state * state) {
    return state ? static_cast<int>(state->scores.size()) : 0;
}

const struct gliner_score * gliner_get_scores(const struct gliner_state * state) {
    return state && !state->scores.empty() ? state->scores.data() : nullptr;
}

} // extern "C"
