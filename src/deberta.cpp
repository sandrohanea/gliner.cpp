#include "internal.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gliner {

void validate_deberta_metadata(const GgufFile & file) {
    if (file.string("gliner.architecture") != "span") {
        throw std::runtime_error("Expected GLiNER2.5-Decide's span checkpoint");
    }
    if (file.string("gliner.token_pooling") != "first") {
        throw std::runtime_error("Only first-subword token pooling is supported");
    }
    const int hidden = file.u32("deberta.hidden_size");
    const int heads = file.u32("deberta.num_attention_heads");
    if (hidden <= 0 || heads <= 0 || hidden % heads != 0 ||
        file.u32("deberta.num_hidden_layers") <= 0 ||
        file.u32("deberta.intermediate_size") <= 0) {
        throw std::runtime_error("Invalid DeBERTa dimensions");
    }
}

Deberta::Deberta(const GgufFile & file, ggml_context * weights) {
    if (file.u32("gliner.format_version") != 2 || file.string("deberta.variant") != "decide-v1") {
        throw std::runtime_error("Unsupported encoder format or configuration");
    }
    hidden = file.u32("deberta.hidden_size");
    heads = file.u32("deberta.num_attention_heads");
    layers = file.u32("deberta.num_hidden_layers");
    intermediate = file.u32("deberta.intermediate_size");
    vocab = file.u32("deberta.vocab_size");
    buckets = file.u32("deberta.position_buckets");
    max_relative = file.u32("deberta.max_relative_positions");
    epsilon = file.f32("deberta.layer_norm_eps");
    if (hidden > 16384 || intermediate > 65536 || layers > 256 || vocab <= 0 ||
        buckets < 4 || buckets > 16384 || buckets % 2 || max_relative <= buckets / 2 + 1 || epsilon <= 0) {
        throw std::runtime_error("Invalid DeBERTa configuration");
    }
    auto add = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        tensors_.emplace(name, file.tensor(weights, "encoder." + name, shape));
    };
    add("embeddings.word_embeddings.weight", {hidden, vocab});
    add("encoder.rel_embeddings.weight", {hidden, 2 * buckets});
    for (const std::string name : {"embeddings.LayerNorm", "encoder.LayerNorm"}) {
        add(name + ".weight", {hidden});
        add(name + ".bias", {hidden});
    }
    for (int i = 0; i < layers; ++i) {
        const std::string prefix = "encoder.layer." + std::to_string(i) + ".";
        for (const std::string name : {"attention.self.query_proj", "attention.self.key_proj",
                                       "attention.self.value_proj", "attention.output.dense"}) {
            add(prefix + name + ".weight", {hidden, hidden});
            add(prefix + name + ".bias", {hidden});
        }
        add(prefix + "intermediate.dense.weight", {hidden, intermediate});
        add(prefix + "intermediate.dense.bias", {intermediate});
        add(prefix + "output.dense.weight", {intermediate, hidden});
        add(prefix + "output.dense.bias", {hidden});
        for (const std::string name : {"attention.output.LayerNorm", "output.LayerNorm"}) {
            add(prefix + name + ".weight", {hidden});
            add(prefix + name + ".bias", {hidden});
        }
    }
}

namespace {
ggml_tensor * f32(ggml_context * ctx, ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}
}

ggml_tensor * Deberta::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul_mat(ctx, tensors_.at(name + ".weight"), x),
                    f32(ctx, tensors_.at(name + ".bias")));
}

ggml_tensor * Deberta::norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, f32(ctx, x), epsilon),
                                 f32(ctx, tensors_.at(name + ".weight"))),
                    f32(ctx, tensors_.at(name + ".bias")));
}

void Deberta::relative_indices(int tokens, std::vector<int32_t> & c2p, std::vector<int32_t> & p2c) const {
    const int64_t table_size = static_cast<int64_t>(2 * buckets) * tokens * heads;
    if (table_size > std::numeric_limits<int32_t>::max()) throw std::invalid_argument("Relative position index overflow");
    const size_t count = static_cast<size_t>(tokens) * tokens * heads;
    c2p.resize(count);
    p2c.resize(count);
    const int mid = buckets / 2;
    for (int h = 0; h < heads; ++h) {
        for (int q = 0; q < tokens; ++q) {
            for (int k = 0; k < tokens; ++k) {
                const int distance = q - k;
                int relative = distance;
                if (std::abs(distance) > mid) {
                    // Match Transformers' float32 log-bucketing, including ceil at boundaries.
                    const float log_position = std::ceil(
                        std::log(static_cast<float>(std::abs(distance)) / mid) /
                        std::log(static_cast<float>(max_relative - 1) / mid) * (mid - 1)) + mid;
                    relative = static_cast<int>(log_position) * (distance < 0 ? -1 : 1);
                }
                relative = std::clamp(relative + buckets, 0, 2 * buckets - 1);
                const size_t at = (static_cast<size_t>(h) * tokens + q) * tokens + k;
                c2p[at] = relative + 2 * buckets * (q + tokens * h);
                // p2c gathers with the opposite relative position, then transposes q/k.
                p2c[at] = relative + 2 * buckets * (k + tokens * h);
            }
        }
    }
}

ggml_tensor * Deberta::build(ggml_context * ctx, ggml_tensor * ids,
                            ggml_tensor * c2p_indices, ggml_tensor * p2c_indices,
                            std::vector<ggml_tensor *> & hidden_states) const {
    const int64_t tokens = ids->ne[0];
    const int64_t dim = hidden / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim * 3));
    auto split_heads = [&](ggml_tensor * x) {
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x, dim, heads, x->ne[1]), 0, 2, 1, 3));
    };
    auto gather = [&](ggml_tensor * scores, ggml_tensor * indices) {
        auto * flat = ggml_reshape_2d(ctx, scores, 1, ggml_nelements(scores));
        return ggml_reshape_3d(ctx, ggml_get_rows(ctx, flat, indices), tokens, tokens, heads);
    };
    auto * x = norm(ctx, ggml_get_rows(ctx, tensors_.at("embeddings.word_embeddings.weight"), ids),
                    "embeddings.LayerNorm");
    hidden_states.push_back(x);
    auto * relative = norm(ctx, tensors_.at("encoder.rel_embeddings.weight"), "encoder.LayerNorm");
    for (int i = 0; i < layers; ++i) {
        const std::string prefix = "encoder.layer." + std::to_string(i) + ".";
        auto * q = split_heads(linear(ctx, x, prefix + "attention.self.query_proj"));
        auto * k = split_heads(linear(ctx, x, prefix + "attention.self.key_proj"));
        auto * v = split_heads(linear(ctx, x, prefix + "attention.self.value_proj"));
        auto * pq = split_heads(linear(ctx, relative, prefix + "attention.self.query_proj"));
        auto * pk = split_heads(linear(ctx, relative, prefix + "attention.self.key_proj"));
        auto * content = ggml_mul_mat(ctx, ggml_scale(ctx, k, scale), q);
        auto * c2p = ggml_scale(ctx, gather(ggml_mul_mat(ctx, pk, q), c2p_indices), scale);
        auto * p2c = ggml_scale(ctx, gather(ggml_mul_mat(ctx, pq, k), p2c_indices), scale);
        auto * probs = ggml_soft_max(ctx, ggml_add(ctx, content, ggml_add(ctx, c2p, p2c)));
        auto * attended = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), probs);
        attended = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), hidden, tokens);
        auto * attention = norm(ctx, ggml_add(ctx, x, linear(ctx, attended, prefix + "attention.output.dense")),
                                prefix + "attention.output.LayerNorm");
        auto * ff = ggml_gelu_erf(ctx, linear(ctx, attention, prefix + "intermediate.dense"));
        x = norm(ctx, ggml_add(ctx, attention, linear(ctx, ff, prefix + "output.dense")),
                 prefix + "output.LayerNorm");
        hidden_states.push_back(x);
    }
    return x;
}

} // namespace gliner
