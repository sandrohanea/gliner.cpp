#include "internal.h"

#include <stdexcept>

namespace gliner {
namespace {
ggml_tensor * f32(ggml_context * ctx, ggml_tensor * tensor) {
    return tensor->type == GGML_TYPE_F32 ? tensor : ggml_cast(ctx, tensor, GGML_TYPE_F32);
}
ggml_tensor * multiply(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    auto * out = ggml_mul_mat(ctx, a, b);
#ifndef GLINER_METAL_FAST_MATH
    if (!ggml_prec_set_acc(out, GGML_PREC_F32) || !ggml_prec_set_src(out, GGML_PREC_F32, 1)) {
        throw std::runtime_error("GGML cannot preserve F32 span precision");
    }
#endif
    return out;
}
}

SpanHead::SpanHead(const GgufFile & file, ggml_context * weights) {
    if (file.string("gliner.span_mode") != "markerV0" || file.string("gliner.counting_layer") != "count_lstm") {
        throw std::runtime_error("Only markerV0 spans with count_lstm conditioning are supported");
    }
    hidden = file.u32("deberta.hidden_size");
    max_width = file.u32("gliner.max_width");
    if (max_width < 1 || max_width > 128) throw std::runtime_error("Invalid span max_width");
    auto add = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        tensors_.emplace(name, file.tensor(weights, name, shape));
    };
    auto linear = [&](const std::string & name, int input, int output) {
        add(name + ".weight", {input, output});
        add(name + ".bias", {output});
    };
    const std::string prefix = "span_rep.span_rep_layer.";
    for (const std::string name : {"project_start", "project_end", "out_project"}) {
        linear(prefix + name + ".0", name == "out_project" ? 2 * hidden : hidden, 4 * hidden);
        linear(prefix + name + ".3", 4 * hidden, hidden);
    }
    linear("count_pred.0", hidden, 2 * hidden);
    linear("count_pred.2", 2 * hidden, 20);
    add("count_embed.pos_embedding.weight", {hidden, 20});
    add("count_embed.gru.weight_ih_l0", {hidden, 3 * hidden});
    add("count_embed.gru.weight_hh_l0", {hidden, 3 * hidden});
    add("count_embed.gru.bias_ih_l0", {3 * hidden});
    add("count_embed.gru.bias_hh_l0", {3 * hidden});
    linear("count_embed.projector.0", 2 * hidden, 4 * hidden);
    linear("count_embed.projector.2", 4 * hidden, hidden);
}

ggml_tensor * SpanHead::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, multiply(ctx, tensors_.at(name + ".weight"), x), f32(ctx, tensors_.at(name + ".bias")));
}

ggml_tensor * SpanHead::projection(ggml_context * ctx, ggml_tensor * x, const std::string & name,
                                  const char * final_layer) const {
    return linear(ctx, ggml_relu(ctx, linear(ctx, x, name + ".0")), name + final_layer);
}

ggml_tensor * SpanHead::build(ggml_context * ctx, ggml_tensor * encoded, ggml_tensor * words,
                             ggml_tensor * labels, ggml_tensor * prompt, ggml_tensor * starts,
                             ggml_tensor * ends, ggml_tensor * count_index, ggml_tensor *& count_logits) const {
    count_logits = count(ctx, ggml_get_rows(ctx, encoded, prompt));
    return score(ctx, ggml_get_rows(ctx, encoded, words), ggml_get_rows(ctx, encoded, labels), starts, ends, count_index);
}

ggml_tensor * SpanHead::count(ggml_context * ctx, ggml_tensor * prompt_states) const {
    return projection(ctx, prompt_states, "count_pred", ".2");
}

ggml_tensor * SpanHead::score(ggml_context * ctx, ggml_tensor * word_states, ggml_tensor * queries,
                             ggml_tensor * starts, ggml_tensor * ends, ggml_tensor * count_indices) const {
    const std::string prefix = "span_rep.span_rep_layer.";
    auto * start_states = projection(ctx, word_states, prefix + "project_start", ".3");
    auto * end_states = projection(ctx, word_states, prefix + "project_end", ".3");
    auto * endpoints = ggml_relu(ctx, ggml_concat(ctx, ggml_get_rows(ctx, start_states, starts),
                                                  ggml_get_rows(ctx, end_states, ends), 0));
    auto * spans = projection(ctx, endpoints, prefix + "out_project", ".3");
    auto * positions = ggml_get_rows(ctx, tensors_.at("count_embed.pos_embedding.weight"), count_indices);
    auto * gi_all = ggml_add(ctx, multiply(ctx, tensors_.at("count_embed.gru.weight_ih_l0"), positions),
                            f32(ctx, tensors_.at("count_embed.gru.bias_ih_l0")));
    auto gate = [&](ggml_tensor * tensor, int index) {
        return ggml_view_2d(ctx, tensor, hidden, tensor->ne[1], tensor->nb[1],
                            static_cast<size_t>(index) * hidden * sizeof(float));
    };
    auto * h = queries;
    ggml_tensor * sequence = nullptr;
    for (int64_t slot = 0; slot < count_indices->ne[0]; ++slot) {
        auto * gi = ggml_view_2d(ctx, gi_all, 3 * hidden, 1, gi_all->nb[1], static_cast<size_t>(slot) * gi_all->nb[1]);
        auto * gh = ggml_add(ctx, multiply(ctx, tensors_.at("count_embed.gru.weight_hh_l0"), h),
                            f32(ctx, tensors_.at("count_embed.gru.bias_hh_l0")));
        auto * r = ggml_sigmoid(ctx, ggml_add(ctx, gate(gh, 0), gate(gi, 0)));
        auto * z = ggml_sigmoid(ctx, ggml_add(ctx, gate(gh, 1), gate(gi, 1)));
        auto * n = ggml_tanh(ctx, ggml_add(ctx, ggml_mul(ctx, r, gate(gh, 2)), gate(gi, 2)));
        h = ggml_add(ctx, ggml_mul(ctx, ggml_sub(ctx, ggml_fill(ctx, z, 1.0f), z), n), ggml_mul(ctx, z, h));
        sequence = sequence ? ggml_concat(ctx, sequence, h, 1) : h;
    }
    auto * projected = projection(ctx, ggml_concat(ctx, sequence, ggml_repeat(ctx, queries, sequence), 0),
                                 "count_embed.projector", ".2");
    return multiply(ctx, spans, projected);
}
} // namespace gliner
