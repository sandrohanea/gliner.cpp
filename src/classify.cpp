#include "gliner/gliner.h"
#include "internal.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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

ggml_backend_dev_t select_device() {
    static const bool loaded = [] {
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        const char * registry = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        if ((std::strcmp(GLINER_COMPILED_BACKEND, "cpu") == 0 && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) ||
            (std::strcmp(GLINER_COMPILED_BACKEND, "cuda") == 0 && std::strcmp(registry, "CUDA") == 0) ||
            (std::strcmp(GLINER_COMPILED_BACKEND, "metal") == 0 &&
             (std::strcmp(registry, "MTL") == 0 || std::strcmp(registry, "Metal") == 0))) {
            return device;
        }
    }
    throw std::runtime_error(std::string("Built for ") + GLINER_COMPILED_BACKEND +
        ", but no matching GGML device is available; check drivers or rebuild for CPU");
}

float sigmoid(float x) {
    if (x >= 0) return 1.f / (1.f + std::exp(-x));
    const float value = std::exp(x);
    return value / (1.f + value);
}

} // namespace

struct gliner_context {
    ggml_backend_dev_t device;
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
    std::string architecture;
    std::unique_ptr<gliner::Tokenizer> tokenizer;
    std::unique_ptr<gliner::Deberta> encoder;
    std::unique_ptr<gliner::SpanHead> span_head;

    explicit gliner_context(gliner::ModelReader & reader)
        : device(select_device()), file(reader),
          weights(ggml_tensor_overhead() * static_cast<size_t>(file.tensor_count()) + 1024 * 1024) {
        gliner::validate_deberta_metadata(file);
        architecture = file.string("gliner.architecture");
        hidden = file.u32("deberta.hidden_size");
        layers = file.u32("deberta.num_hidden_layers");
        tensors = file.tensor_count();
        backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) throw std::runtime_error(std::string("Cannot initialize device: ") + ggml_backend_dev_name(device));
        try {
            if (hidden > 16384) throw std::runtime_error("Unsupported hidden_size");
            w1 = file.tensor(weights.value, "classifier.0.weight", {hidden, 2 * hidden});
            b1 = file.tensor(weights.value, "classifier.0.bias", {2 * hidden});
            const int output_index = file.has("gliner.classifier_output_index") ? file.u32("gliner.classifier_output_index") : 2;
            if ((architecture == "span" && output_index != 2) ||
                (architecture == "boundary" && output_index != 2 && output_index != 3)) {
                throw std::runtime_error("Unsupported classifier output layer index");
            }
            const auto output = "classifier." + std::to_string(output_index);
            w2 = file.tensor(weights.value, output + ".weight", {2 * hidden, 1});
            b2 = file.tensor(weights.value, output + ".bias", {1});
            if (file.has("gliner.format_version")) {
                encoder = std::make_unique<gliner::Deberta>(file, weights.value);
                tokenizer = std::make_unique<gliner::Tokenizer>(file);
                if (file.has("gliner.span_mode")) span_head = std::make_unique<gliner::SpanHead>(file, weights.value);
            }
            weight_buffer = ggml_backend_alloc_ctx_tensors(weights.value, backend);
            if (!weight_buffer) throw std::runtime_error(std::string("Cannot allocate model weights on ") + ggml_backend_dev_name(device));
            ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            file.load_weights(weights.value, reader);
            ggml_backend_synchronize(backend);
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
    std::vector<gliner_span> spans;
    std::vector<std::string> span_texts;
    std::vector<int32_t> span_starts, span_ends;
    std::vector<float> span_logits, count_logits;
    gliner_span_scores span_scores = {};
    std::vector<gliner_record> records;
    std::vector<gliner_span> record_spans;
    gliner_record_scores record_scores = {};
    gliner_eval_callback callback = nullptr;
    void * callback_data = nullptr;

    void clear() {
        scores.clear();
        tokens.ids.clear();
        tokens.markers.clear();
        tokens.tasks.clear();
        tokens.words.clear();
        tokens.word_positions.clear();
        label_states.clear();
        spans.clear();
        span_texts.clear();
        span_starts.clear();
        span_ends.clear();
        span_logits.clear();
        count_logits.clear();
        span_scores = {};
        records.clear();
        record_spans.clear();
        record_scores = {};
    }

    explicit gliner_state(const gliner_context * ctx) : owner(ctx) {
        backend = ggml_backend_dev_init(ctx->device, nullptr);
        if (!backend) throw std::runtime_error(std::string("Cannot initialize state on ") + ggml_backend_dev_name(ctx->device));
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
#ifndef GLINER_METAL_FAST_MATH
    if (!ggml_prec_set_acc(h, GGML_PREC_F32) || !ggml_prec_set_src(h, GGML_PREC_F32, 1)) {
        throw std::runtime_error("GGML cannot preserve F32 classifier precision");
    }
#endif
    auto * b1 = ctx->b1->type == GGML_TYPE_F32 ? ctx->b1 : ggml_cast(eval, ctx->b1, GGML_TYPE_F32);
    h = ggml_relu(eval, ggml_add(eval, h, b1));
    auto * out = ggml_mul_mat(eval, ctx->w2, h);
#ifndef GLINER_METAL_FAST_MATH
    if (!ggml_prec_set_acc(out, GGML_PREC_F32) || !ggml_prec_set_src(out, GGML_PREC_F32, 1)) {
        throw std::runtime_error("GGML cannot preserve F32 classifier precision");
    }
#endif
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
    if (ggml_backend_dev_type(state->owner->device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        const auto registry = ggml_backend_dev_backend_reg(state->owner->device);
        const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
        if (set_threads) set_threads(state->backend, n_threads);
        else if (n_threads != 1) {
            set_error("This ggml CPU backend cannot set the thread count");
            return false;
        }
    }
    const auto status = ggml_backend_graph_compute(state->backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        char message[512];
        std::snprintf(message, sizeof(message), "ggml inference graph failed on %s (status %d)",
                      ggml_backend_dev_name(state->owner->device), static_cast<int>(status));
        set_error(message);
        return false;
    }
    return true;
}

bool allocate_graph(gliner_state * state, ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT && ggml_backend_dev_type(state->owner->device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            if (!ggml_prec_set_acc(node, GGML_PREC_F32) || !ggml_prec_set_src(node, GGML_PREC_F32, 1)) {
                set_error("Cannot request F32 matrix inputs and accumulation");
                return false;
            }
        }
        if (!ggml_backend_dev_supports_op(state->owner->device, node)) {
            char message[512];
            std::snprintf(message, sizeof(message), "Device %s does not support graph operation %s (%s); no CPU fallback",
                          ggml_backend_dev_name(state->owner->device), ggml_op_desc(node), node->name);
            set_error(message);
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(state->allocator, graph)) {
        char message[512];
        std::snprintf(message, sizeof(message), "Cannot allocate inference graph on %s",
                      ggml_backend_dev_name(state->owner->device));
        set_error(message);
        return false;
    }
    return true;
}

void notify_layers(gliner_state * state, const std::vector<ggml_tensor *> & hidden, int n_tokens, int width) {
    if (!state->callback) return;
    std::vector<float> values(static_cast<size_t>(n_tokens) * width);
    for (size_t i = 0; i < hidden.size(); ++i) {
        ggml_backend_tensor_get(hidden[i], values.data(), 0, values.size() * sizeof(float));
        state->callback(static_cast<int>(i), values.data(), n_tokens, width, state->callback_data);
    }
}

struct EncoderGraph {
    std::vector<int32_t> c2p, p2c;
    size_t capacity;
    GraphContext eval;
    ggml_tensor * ids;
    ggml_tensor * c2p_indices;
    ggml_tensor * p2c_indices;
    ggml_tensor * encoded;
    std::vector<ggml_tensor *> hidden;

    EncoderGraph(const gliner_context * ctx, const gliner::Tokenized & tokens, bool capture)
        : capacity(static_cast<size_t>(ctx->layers + 1) * 160 + 512),
          eval(ggml_tensor_overhead() * capacity * 2 + ggml_graph_overhead_custom(capacity, false)) {
        ctx->encoder->relative_indices(static_cast<int>(tokens.ids.size()), c2p, p2c);
        ids = input(static_cast<int64_t>(tokens.ids.size()));
        c2p_indices = input(static_cast<int64_t>(c2p.size()));
        p2c_indices = input(static_cast<int64_t>(p2c.size()));
        encoded = ctx->encoder->build(eval.value, ids, c2p_indices, p2c_indices, hidden);
        if (capture) for (auto * layer : hidden) ggml_set_output(layer);
    }
    ggml_tensor * input(int64_t size) {
        auto * tensor = ggml_new_tensor_1d(eval.value, GGML_TYPE_I32, size);
        ggml_set_input(tensor);
        return tensor;
    }
    void upload(const gliner::Tokenized & tokens) {
        ggml_backend_tensor_set(ids, tokens.ids.data(), 0, tokens.ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(c2p_indices, c2p.data(), 0, c2p.size() * sizeof(int32_t));
        ggml_backend_tensor_set(p2c_indices, p2c.data(), 0, p2c.size() * sizeof(int32_t));
    }
};

void candidate_indices(int words, int max_width, std::vector<int32_t> & starts, std::vector<int32_t> & ends) {
    for (int start = 0; start < words; ++start) {
        for (int width = 1; width <= max_width && start + width <= words; ++width) {
            starts.push_back(start);
            ends.push_back(start + width - 1);
        }
    }
}

std::vector<gliner_span> select_spans(const gliner::Tokenized & tokens, const std::vector<int32_t> & starts,
                                     const std::vector<int32_t> & ends, const float * logits, int label,
                                     float threshold, bool allow_overlap, int limit) {
    std::vector<gliner_span> candidates, selected;
    for (size_t i = 0; i < starts.size(); ++i) {
        const auto & first = tokens.words[starts[i]];
        const auto & last = tokens.words[ends[i]];
        // Drop synthetic suffix tokens; offsets for a suffix inside a URL are clipped.
        if (first.start == first.end || last.start == last.end) continue;
        const float probability = sigmoid(logits[i]);
        if (probability >= threshold) candidates.push_back({label, first.start, last.end, logits[i], probability, nullptr});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const gliner_span & a, const gliner_span & b) {
        if (a.probability != b.probability) return a.probability > b.probability;
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end;
    });
    for (const auto & candidate : candidates) {
        if (!allow_overlap && std::any_of(selected.begin(), selected.end(), [&](const gliner_span & other) {
            return candidate.start < other.end && other.start < candidate.end;
        })) continue;
        selected.push_back(candidate);
        if (limit && selected.size() == static_cast<size_t>(limit)) break;
    }
    return selected;
}

std::vector<float> download_finite(ggml_tensor * tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("Model produced nonfinite extraction logits");
    return values;
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
        std::ifstream input(std::filesystem::u8path(gguf_path), std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot open GGUF file");
        const auto size = input.tellg();
        if (size < 0) throw std::runtime_error("Cannot determine GGUF file size");
        input.seekg(0);
        if (!input) throw std::runtime_error("Cannot seek to GGUF header");
        gliner_model_loader loader = {
            &input,
            [](void * ctx, void * output, size_t count) -> size_t {
                auto & stream = *static_cast<std::ifstream *>(ctx);
                stream.read(static_cast<char *>(output), static_cast<std::streamsize>(count));
                return static_cast<size_t>(stream.gcount());
            },
            [](void * ctx) -> bool { return static_cast<std::ifstream *>(ctx)->eof(); },
            [](void *) {}
        };
        gliner::ModelReader reader(loader, static_cast<uint64_t>(size));
        return new gliner_context(reader);
    } catch (const std::exception & error) {
        set_error(error.what());
        return nullptr;
    } catch (...) {
        set_error("Unknown model loading error");
        return nullptr;
    }
}

struct gliner_context * gliner_init_from_buffer(const void * buffer, size_t buffer_size) {
    clear_error();
    if (!buffer || !buffer_size) {
        set_error("A nonempty GGUF buffer is required");
        return nullptr;
    }
    try {
        gliner::ModelReader reader(buffer, buffer_size);
        return new gliner_context(reader);
    } catch (const std::exception & error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown buffer loading error");
    }
    return nullptr;
}

struct gliner_context * gliner_init(struct gliner_model_loader * loader) {
    clear_error();
    if (!loader || !loader->read || !loader->eof || !loader->close) {
        set_error("Model loader requires read, eof and close callbacks");
        return nullptr;
    }
    const auto callbacks = *loader;
    std::unique_ptr<gliner_context> model;
    try {
        gliner::ModelReader reader(callbacks);
        model = std::make_unique<gliner_context>(reader);
    } catch (const std::exception & error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown stream loading error");
    }
    try {
        callbacks.close(callbacks.context);
    } catch (const std::exception & error) {
        model.reset();
        set_error(error.what());
    } catch (...) {
        model.reset();
        set_error("Model loader close callback failed");
    }
    return model.release();
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
int gliner_model_supports_spans(const struct gliner_context * ctx) { return ctx && ctx->span_head ? 1 : 0; }
const char * gliner_model_architecture(const struct gliner_context * ctx) { return ctx ? ctx->architecture.c_str() : nullptr; }
const char * gliner_model_backend_name(const struct gliner_context * ctx) {
    return ctx ? GLINER_COMPILED_BACKEND : nullptr;
}
const char * gliner_build_backend(void) { return GLINER_COMPILED_BACKEND; }
const char * gliner_model_device_name(const struct gliner_context * ctx) {
    return ctx ? ggml_backend_dev_name(ctx->device) : nullptr;
}

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
        if (!allocate_graph(state, graph)) return GLINER_STATUS_BACKEND_ERROR;
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

struct gliner_batch_params gliner_default_batch_params(void) {
    const auto params = gliner_default_text_params();
    return {params.n_threads, params.max_tokens, params.max_words};
}

void gliner_set_eval_callback(struct gliner_state * state, gliner_eval_callback callback, void * user_data) {
    if (state) { state->callback = callback; state->callback_data = user_data; }
}

int gliner_classify_text(const struct gliner_context * ctx, struct gliner_state * state,
                        const char * text, const char * task, const char * const * labels,
                        int n_labels, const struct gliner_text_params * params) {
    const auto options = params ? *params : gliner_default_text_params();
    const gliner_classification_task entry = {task, labels, n_labels, options.prompt, options.label_descriptions};
    const gliner_batch_params batch_options = {options.n_threads, options.max_tokens, options.max_words};
    return gliner_classify_text_batch(ctx, state, text, &entry, 1, &batch_options);
}

int gliner_classify_text_batch(const struct gliner_context * ctx, struct gliner_state * state,
                              const char * text, const struct gliner_classification_task * tasks,
                              int n_tasks, const struct gliner_batch_params * params) {
    clear_error();
    if (state) state->clear();
    const auto options = params ? *params : gliner_default_batch_params();
    if (!ctx || !state || state->owner != ctx || !text || !tasks || n_tasks <= 0 ||
        options.n_threads <= 0 || options.max_words < 0 || options.max_tokens <= 0 ||
        options.max_tokens > 4096 || n_tasks > options.max_tokens) {
        set_error("Invalid text batch arguments (max_tokens must be 1..4096)");
        return GLINER_STATUS_INVALID_ARGUMENT;
    }
    if (!ctx->encoder || !ctx->tokenizer) {
        set_error("This legacy GGUF supports only label states; reconvert the checkpoint for text inference");
        return GLINER_STATUS_MODEL_ERROR;
    }
    try {
        int n_labels = 0;
        std::vector<gliner::ClassificationTask> schema;
        schema.reserve(static_cast<size_t>(n_tasks));
        for (int t = 0; t < n_tasks; ++t) {
            const auto & task = tasks[t];
            if (!task.task || !task.labels || task.n_labels <= 0 || task.n_labels > options.max_tokens - n_labels) {
                throw std::invalid_argument("Each task requires a name and labels; total labels must fit max_tokens");
            }
            n_labels += task.n_labels;
            gliner::ClassificationTask entry;
            entry.name = task.task;
            entry.prompt = task.prompt ? task.prompt : "";
            for (int i = 0; i < task.n_labels; ++i) {
                if (!task.labels[i]) throw std::invalid_argument("NULL label");
                entry.labels.emplace_back(task.labels[i]);
                if (task.label_descriptions && task.label_descriptions[i]) entry.descriptions.emplace_back(task.label_descriptions[i]);
                else entry.descriptions.emplace_back(std::nullopt);
            }
            schema.push_back(std::move(entry));
        }
        auto tokens = ctx->tokenizer->encode(text, schema, options.max_words, options.max_tokens);
        const int n_tokens = static_cast<int>(tokens.ids.size());
        EncoderGraph graph_data(ctx, tokens, state->callback != nullptr);
        auto * eval = graph_data.eval.value;
        auto * markers = graph_data.input(n_labels);
        auto * label_states = ggml_get_rows(eval, graph_data.encoded, markers);
        ggml_set_output(label_states);
        auto * out = classification_graph(eval, ctx, label_states);
        auto * graph = ggml_new_graph_custom(eval, graph_data.capacity, false);
        ggml_build_forward_expand(graph, out);
        if (!allocate_graph(state, graph)) return GLINER_STATUS_BACKEND_ERROR;
        graph_data.upload(tokens);
        ggml_backend_tensor_set(markers, tokens.markers.data(), 0, tokens.markers.size() * sizeof(int32_t));
        if (!compute(state, graph, options.n_threads)) return GLINER_STATUS_BACKEND_ERROR;
        collect_scores(state, out, static_cast<size_t>(n_labels));
        state->label_states.resize(static_cast<size_t>(n_labels) * ctx->hidden);
        ggml_backend_tensor_get(label_states, state->label_states.data(), 0, state->label_states.size() * sizeof(float));
        notify_layers(state, graph_data.hidden, n_tokens, ctx->hidden);
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

struct gliner_span_params gliner_default_span_params(void) {
    return {1, 512, 0, 0.5f, 0, 0};
}

int gliner_extract_spans(const struct gliner_context * ctx, struct gliner_state * state,
                         const char * text, const struct gliner_span_label * labels,
                         int n_labels, const struct gliner_span_params * params) {
    clear_error();
    if (state) state->clear();
    const auto options = params ? *params : gliner_default_span_params();
    if (!ctx || !state || state->owner != ctx || !text || !labels || n_labels <= 0 ||
        options.n_threads <= 0 || options.max_tokens <= 0 || options.max_tokens > 4096 ||
        n_labels > options.max_tokens || options.max_words < 0 || options.max_spans_per_label < 0 ||
        (options.allow_overlap != 0 && options.allow_overlap != 1) ||
        !std::isfinite(options.threshold) || options.threshold < 0 || options.threshold > 1) {
        set_error("Invalid span extraction arguments");
        return GLINER_STATUS_INVALID_ARGUMENT;
    }
    if (!ctx->span_head) {
        set_error(ctx->architecture == "boundary" ? "Boundary span extraction is not implemented; this model supports classification only" :
                  "Model has no supported span metadata; reconvert a markerV0/count_lstm checkpoint");
        return GLINER_STATUS_MODEL_ERROR;
    }
    try {
        const std::string source(text);
        gliner::ClassificationTask schema;
        schema.name = "entities";
        for (int i = 0; i < n_labels; ++i) {
            if (!labels[i].name) throw std::invalid_argument("NULL extraction label");
            schema.labels.emplace_back(labels[i].name);
            if (labels[i].description) schema.descriptions.emplace_back(labels[i].description);
            else schema.descriptions.emplace_back(std::nullopt);
        }
        auto tokens = ctx->tokenizer->encode(source, {schema}, options.max_words, options.max_tokens, gliner::SchemaKind::Entities);
        const int n_words = static_cast<int>(tokens.words.size());
        std::vector<int32_t> starts, ends;
        candidate_indices(n_words, ctx->span_head->max_width, starts, ends);
        EncoderGraph data(ctx, tokens, state->callback != nullptr);
        auto * eval = data.eval.value;
        auto * word_ids = data.input(n_words);
        auto * label_ids = data.input(n_labels);
        auto * prompt = data.input(1);
        auto * start_ids = data.input(static_cast<int64_t>(starts.size()));
        auto * end_ids = data.input(static_cast<int64_t>(ends.size()));
        auto * count_index = data.input(1);
        ggml_tensor * count_out = nullptr;
        auto * out = ctx->span_head->build(eval, data.encoded, word_ids, label_ids, prompt, start_ids, end_ids, count_index, count_out);
        ggml_set_output(out);
        ggml_set_output(count_out);
        auto * graph = ggml_new_graph_custom(eval, data.capacity, false);
        ggml_build_forward_expand(graph, count_out);
        ggml_build_forward_expand(graph, out);
        if (!allocate_graph(state, graph)) return GLINER_STATUS_BACKEND_ERROR;
        data.upload(tokens);
        ggml_backend_tensor_set(word_ids, tokens.word_positions.data(), 0, tokens.word_positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(label_ids, tokens.markers.data(), 0, tokens.markers.size() * sizeof(int32_t));
        ggml_backend_tensor_set(prompt, &tokens.prompt_position, 0, sizeof(int32_t));
        ggml_backend_tensor_set(start_ids, starts.data(), 0, starts.size() * sizeof(int32_t));
        ggml_backend_tensor_set(end_ids, ends.data(), 0, ends.size() * sizeof(int32_t));
        const int32_t zero = 0;
        ggml_backend_tensor_set(count_index, &zero, 0, sizeof(zero));
        if (!compute(state, graph, options.n_threads)) return GLINER_STATUS_BACKEND_ERROR;
        state->span_logits = download_finite(out);
        state->count_logits = download_finite(count_out);
        const int predicted = static_cast<int>(std::max_element(state->count_logits.begin(), state->count_logits.end()) - state->count_logits.begin());
        if (predicted > 0) {
            for (int label = 0; label < n_labels; ++label) {
                const auto selected = select_spans(tokens, starts, ends,
                    state->span_logits.data() + static_cast<size_t>(label) * starts.size(), label,
                    options.threshold, options.allow_overlap != 0, options.max_spans_per_label);
                state->spans.insert(state->spans.end(), selected.begin(), selected.end());
            }
        }
        state->span_texts.reserve(state->spans.size());
        for (auto & span : state->spans) {
            state->span_texts.push_back(source.substr(span.start, span.end - span.start));
            span.text = state->span_texts.back().c_str();
        }
        notify_layers(state, data.hidden, static_cast<int>(tokens.ids.size()), ctx->hidden);
        tokens.tasks.clear();
        state->tokens = std::move(tokens);
        state->span_starts = std::move(starts);
        state->span_ends = std::move(ends);
        for (auto & end : state->span_ends) ++end;
        state->span_scores = {n_labels, static_cast<int>(state->span_starts.size()), n_words, ctx->span_head->max_width,
                             predicted, state->tokens.markers.data(), state->tokens.word_positions.data(),
                             state->span_starts.data(), state->span_ends.data(), state->span_logits.data(), state->count_logits.data()};
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
        set_error("Unknown span extraction error");
        return GLINER_STATUS_MODEL_ERROR;
    }
}

struct gliner_record_params gliner_default_record_params(void) {
    return {1, 512, 0, 0.5f, 19, 0, 0};
}

int gliner_extract_records(const struct gliner_context * ctx, struct gliner_state * state,
                           const char * text, const char * structure,
                           const struct gliner_record_field * fields, int n_fields,
                           const struct gliner_record_params * params) {
    clear_error();
    if (state) state->clear();
    const auto options = params ? *params : gliner_default_record_params();
    if (!ctx || !state || state->owner != ctx || !text || !structure || !*structure || !fields ||
        n_fields <= 0 || options.n_threads <= 0 || options.max_tokens <= 0 || options.max_tokens > 4096 ||
        n_fields > options.max_tokens || options.max_words < 0 || options.max_records < 1 || options.max_records > 19 ||
        options.max_spans_per_field < 0 || (options.allow_overlap != 0 && options.allow_overlap != 1) ||
        !std::isfinite(options.threshold) || options.threshold < 0 || options.threshold > 1 ||
        std::strcmp(structure, "entities") == 0) {
        set_error("Invalid structured extraction arguments (max_records must be 1..19; 'entities' is reserved)");
        return GLINER_STATUS_INVALID_ARGUMENT;
    }
    if (!ctx->span_head) {
        set_error(ctx->architecture == "boundary" ? "Boundary record extraction is not implemented; this model supports classification only" :
                  "Model has no supported span metadata; reconvert a markerV0/count_lstm checkpoint");
        return GLINER_STATUS_MODEL_ERROR;
    }
    try {
        const std::string source(text);
        gliner::ClassificationTask schema;
        schema.name = structure;
        for (int i = 0; i < n_fields; ++i) {
            if (!fields[i].name || (fields[i].type != GLINER_FIELD_LIST && fields[i].type != GLINER_FIELD_SINGLE)) {
                throw std::invalid_argument("Fields require names and a list/single type");
            }
            schema.labels.emplace_back(fields[i].name);
            if (fields[i].description) schema.descriptions.emplace_back(fields[i].description);
            else schema.descriptions.emplace_back(std::nullopt);
        }
        auto tokens = ctx->tokenizer->encode(source, {schema}, options.max_words, options.max_tokens, gliner::SchemaKind::Records);
        const int n_words = static_cast<int>(tokens.words.size());
        std::vector<int32_t> starts, ends;
        candidate_indices(n_words, ctx->span_head->max_width, starts, ends);
        EncoderGraph data(ctx, tokens, state->callback != nullptr);
        auto * word_ids = data.input(n_words);
        auto * field_ids = data.input(n_fields);
        auto * prompt_id = data.input(1);
        auto * words = ggml_get_rows(data.eval.value, data.encoded, word_ids);
        auto * queries = ggml_get_rows(data.eval.value, data.encoded, field_ids);
        auto * count = ctx->span_head->count(data.eval.value, ggml_get_rows(data.eval.value, data.encoded, prompt_id));
        for (auto * output : {words, queries, count}) ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(data.eval.value, data.capacity, false);
        ggml_build_forward_expand(graph, count);
        ggml_build_forward_expand(graph, words);
        ggml_build_forward_expand(graph, queries);
        if (!allocate_graph(state, graph)) return GLINER_STATUS_BACKEND_ERROR;
        data.upload(tokens);
        ggml_backend_tensor_set(word_ids, tokens.word_positions.data(), 0, tokens.word_positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(field_ids, tokens.markers.data(), 0, tokens.markers.size() * sizeof(int32_t));
        ggml_backend_tensor_set(prompt_id, &tokens.prompt_position, 0, sizeof(int32_t));
        if (!compute(state, graph, options.n_threads)) return GLINER_STATUS_BACKEND_ERROR;
        auto count_values = download_finite(count);
        const int predicted = static_cast<int>(std::max_element(count_values.begin(), count_values.end()) - count_values.begin());
        if (predicted > options.max_records) {
            throw std::invalid_argument("Predicted " + std::to_string(predicted) + " records exceeds max_records; no records were truncated");
        }
        notify_layers(state, data.hidden, static_cast<int>(tokens.ids.size()), ctx->hidden);
        std::vector<float> logits;
        if (predicted > 0) {
            // Retain features on the same device while the reusable graph allocator moves to the head graph.
            GraphContext retained(4 * ggml_tensor_overhead() + 1024);
            auto * saved_words = ggml_dup_tensor(retained.value, words);
            auto * saved_queries = ggml_dup_tensor(retained.value, queries);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(retained.value, state->backend), ggml_backend_buffer_free);
            if (!buffer) {
                set_error("Cannot retain encoder features for structured extraction");
                return GLINER_STATUS_BACKEND_ERROR;
            }
            ggml_backend_tensor_copy(words, saved_words);
            ggml_backend_tensor_copy(queries, saved_queries);
            const size_t capacity = 256 + static_cast<size_t>(predicted) * 80;
            GraphContext head(ggml_tensor_overhead() * capacity * 2 + ggml_graph_overhead_custom(capacity, false));
            auto * start_ids = ggml_new_tensor_1d(head.value, GGML_TYPE_I32, static_cast<int64_t>(starts.size()));
            auto * end_ids = ggml_new_tensor_1d(head.value, GGML_TYPE_I32, static_cast<int64_t>(ends.size()));
            auto * indices = ggml_new_tensor_1d(head.value, GGML_TYPE_I32, predicted);
            for (auto * input : {start_ids, end_ids, indices}) ggml_set_input(input);
            auto * out = ctx->span_head->score(head.value, saved_words, saved_queries, start_ids, end_ids, indices);
            ggml_set_output(out);
            auto * head_graph = ggml_new_graph_custom(head.value, capacity, false);
            ggml_build_forward_expand(head_graph, out);
            if (!allocate_graph(state, head_graph)) return GLINER_STATUS_BACKEND_ERROR;
            std::vector<int32_t> steps(static_cast<size_t>(predicted));
            for (int i = 0; i < predicted; ++i) steps[i] = i;
            ggml_backend_tensor_set(start_ids, starts.data(), 0, starts.size() * sizeof(int32_t));
            ggml_backend_tensor_set(end_ids, ends.data(), 0, ends.size() * sizeof(int32_t));
            ggml_backend_tensor_set(indices, steps.data(), 0, steps.size() * sizeof(int32_t));
            if (!compute(state, head_graph, options.n_threads)) return GLINER_STATUS_BACKEND_ERROR;
            logits = download_finite(out);
        }
        for (int slot = 0; slot < predicted; ++slot) {
            const size_t offset = state->record_spans.size();
            for (int field = 0; field < n_fields; ++field) {
                const int limit = fields[field].type == GLINER_FIELD_SINGLE ? 1 : options.max_spans_per_field;
                const size_t base = (static_cast<size_t>(slot) * n_fields + field) * starts.size();
                const auto selected = select_spans(tokens, starts, ends, logits.data() + base, field,
                                                   options.threshold, options.allow_overlap != 0, limit);
                if (selected.size() > static_cast<size_t>(std::numeric_limits<int>::max()) - state->record_spans.size()) {
                    throw std::runtime_error("Too many structured spans");
                }
                state->record_spans.insert(state->record_spans.end(), selected.begin(), selected.end());
            }
            const size_t span_count = state->record_spans.size() - offset;
            if (span_count) state->records.push_back({slot, static_cast<int>(offset), static_cast<int>(span_count)});
        }
        state->span_texts.reserve(state->record_spans.size());
        for (auto & span : state->record_spans) {
            state->span_texts.push_back(source.substr(span.start, span.end - span.start));
            span.text = state->span_texts.back().c_str();
        }
        tokens.tasks.clear();
        state->tokens = std::move(tokens);
        state->span_starts = std::move(starts);
        state->span_ends = std::move(ends);
        for (auto & end : state->span_ends) ++end;
        state->span_logits = std::move(logits);
        state->count_logits = std::move(count_values);
        state->record_scores = {n_fields, static_cast<int>(state->span_starts.size()), n_words, ctx->span_head->max_width, predicted,
                               state->tokens.markers.data(), state->tokens.word_positions.data(), state->span_starts.data(),
                               state->span_ends.data(), state->span_logits.empty() ? nullptr : state->span_logits.data(),
                               state->count_logits.data()};
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
        set_error("Unknown structured extraction error");
        return GLINER_STATUS_MODEL_ERROR;
    }
}

int gliner_n_records(const struct gliner_state * state) {
    return state ? static_cast<int>(state->records.size()) : 0;
}
const struct gliner_record * gliner_get_records(const struct gliner_state * state) {
    return state && !state->records.empty() ? state->records.data() : nullptr;
}
const struct gliner_span * gliner_get_record_spans(const struct gliner_state * state) {
    return state && !state->record_spans.empty() ? state->record_spans.data() : nullptr;
}
const struct gliner_record_scores * gliner_get_record_scores(const struct gliner_state * state) {
    return state && state->record_scores.n_fields > 0 ? &state->record_scores : nullptr;
}

int gliner_n_spans(const struct gliner_state * state) {
    return state ? static_cast<int>(state->spans.size()) : 0;
}
const struct gliner_span * gliner_get_spans(const struct gliner_state * state) {
    return state && !state->spans.empty() ? state->spans.data() : nullptr;
}
const struct gliner_span_scores * gliner_get_span_scores(const struct gliner_state * state) {
    return state && state->span_scores.n_labels > 0 ? &state->span_scores : nullptr;
}

int gliner_n_tasks(const struct gliner_state * state) {
    return state ? static_cast<int>(state->tokens.tasks.size()) : 0;
}
const struct gliner_task_result * gliner_get_task_results(const struct gliner_state * state) {
    return state && !state->tokens.tasks.empty() ? state->tokens.tasks.data() : nullptr;
}

int gliner_n_tokens(const struct gliner_state * state) {
    return state ? static_cast<int>(state->tokens.ids.size()) : 0;
}
const int32_t * gliner_get_token_ids(const struct gliner_state * state) {
    return state && !state->tokens.ids.empty() ? state->tokens.ids.data() : nullptr;
}
const int32_t * gliner_get_label_positions(const struct gliner_state * state) {
    return state && !state->scores.empty() && !state->tokens.markers.empty() ? state->tokens.markers.data() : nullptr;
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
