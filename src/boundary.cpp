#include "internal.h"

#include <cmath>
#include <stdexcept>

namespace gliner {
namespace {
ggml_tensor * f32(ggml_context * ctx, ggml_tensor * x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
}
ggml_tensor * mm(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    auto * out = ggml_mul_mat(ctx, a, b);
#ifndef GLINER_METAL_FAST_MATH
    if (!ggml_prec_set_acc(out, GGML_PREC_F32) || !ggml_prec_set_src(out, GGML_PREC_F32, 1)) {
        throw std::runtime_error("GGML cannot preserve F32 boundary precision");
    }
#endif
    return out;
}
ggml_tensor * part(ggml_context * ctx, ggml_tensor * x, int width, int index) {
    return ggml_cont(ctx, ggml_view_2d(ctx, x, width, x->ne[1], x->nb[1], static_cast<size_t>(index) * width * sizeof(float)));
}
ggml_tensor * transpose(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}
ggml_tensor * gather_columns(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids) {
    return transpose(ctx, ggml_get_rows(ctx, transpose(ctx, x), ids));
}
ggml_tensor * prefix(ggml_context * ctx, ggml_tensor * rows) {
    auto * zeros = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, rows->ne[1]), 0);
    return ggml_concat(ctx, zeros, ggml_cumsum(ctx, rows), 0);
}
}

BoundaryHead::BoundaryHead(const GgufFile & file, ggml_context * weights) {
    if (file.string("gliner.boundary.variant") != "shared-v1") throw std::runtime_error("Unsupported boundary head variant");
    hidden = file.u32("deberta.hidden_size");
    dim = file.u32("gliner.boundary.boundary_dim");
    pair_dim = file.u32("gliner.boundary.pair_dim");
    content_dim = file.u32("gliner.boundary.content_dim");
    heads = file.u32("gliner.boundary.boundary_attention_heads");
    attention_layers = file.u32("gliner.boundary.boundary_attention_layers");
    window = file.u32("gliner.boundary.boundary_attention_window");
    refinement_layers = file.u32("gliner.boundary.boundary_refinement_layers");
    ffn_size = file.u32("gliner.boundary.ffn_size");
    top_k = file.u32("gliner.boundary.pool_boundary_top_k");
    pool_size = file.u32("gliner.boundary.pool_size");
    quota = file.u32("gliner.boundary.min_pool_per_query");
    temperature = file.f32("gliner.boundary.pair_temperature");
    abstention_threshold = file.f32("gliner.boundary.abstention_threshold");
    if (dim < 1 || dim > 4096 || pair_dim < 1 || pair_dim > 4096 || content_dim < 1 || content_dim > 4096 ||
        heads < 1 || heads > 128 || dim % heads || attention_layers > 16 || refinement_layers > 16 || window > 4096 ||
        ffn_size < 1 || ffn_size > 16 * dim || top_k < 1 || top_k > 256 ||
        pool_size < 1 || pool_size > 4096 || quota > 256 || temperature <= 0 || abstention_threshold < 0 || abstention_threshold > 1) {
        throw std::runtime_error("Invalid boundary head dimensions or settings");
    }
    auto add = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        tensors_.emplace(name, file.tensor(weights, "boundary_head." + name, shape));
    };
    auto linear = [&](const std::string & name, int in, int out) {
        add(name + ".weight", {in, out});
        add(name + ".bias", {out});
    };
    auto norm = [&](const std::string & name, int width) {
        add(name + ".weight", {width}); add(name + ".bias", {width});
    };
    for (const std::string side : {"left", "right"}) linear("boundary_encoder." + side + "_projection", hidden, dim);
    for (const std::string side : {"bos", "eos"}) add("boundary_encoder." + side + "_state", {hidden});
    linear("boundary_encoder.output_projection", 2 * dim, dim);
    norm("boundary_encoder.layer_norm", dim);
    for (int i = 0; i < attention_layers; ++i) {
        const auto p = "boundary_encoder.attention_blocks." + std::to_string(i) + ".";
        norm(p + "norm", dim);
        linear(p + "qkv_projection", dim, 3 * dim);
        linear(p + "output_projection", dim, dim);
    }
    for (int i = 0; i < refinement_layers; ++i) {
        const auto p = "boundary_encoder.refinement_blocks." + std::to_string(i) + ".";
        norm(p + "norm", dim);
        linear(p + "input_projection", dim, 2 * ffn_size);
        linear(p + "output_projection", ffn_size, dim);
    }
    for (const std::string side : {"start", "end"}) {
        linear("boundary_query_head." + side + "_boundary_projection", dim, dim);
        linear("boundary_query_head." + side + "_query_projection", hidden, dim);
        linear("shared_pool_builder." + side + "_projection", dim, dim);
        linear("shared_pool_scorer." + side + "_projection", dim, pair_dim);
    }
    linear("boundary_query_head.inside_text_projection", hidden, dim);
    linear("boundary_query_head.inside_query_projection", hidden, dim);
    linear("null_projection", hidden, 1);
    linear("shared_pool_scorer.content_pooler.value_projection", hidden, content_dim);
    norm("shared_pool_scorer.content_pooler.layer_norm", content_dim);
    linear("shared_pool_scorer.content_projection", content_dim, pair_dim);
    linear("shared_pool_scorer.length_projection", 3, pair_dim);
    linear("shared_pool_scorer.prior_projection", 1, pair_dim);
    norm("shared_pool_scorer.candidate_norm", pair_dim);
    linear("shared_pool_scorer.query_projection", hidden, pair_dim);
    linear("shared_pool_scorer.film", pair_dim, 2 * pair_dim);
    linear("shared_pool_scorer.film_output.0", pair_dim, 64);
    linear("shared_pool_scorer.film_output.3", 64, 1);
}

ggml_tensor * BoundaryHead::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, mm(ctx, tensors_.at(name + ".weight"), x), f32(ctx, tensors_.at(name + ".bias")));
}

ggml_tensor * BoundaryHead::norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), f32(ctx, tensors_.at(name + ".weight"))),
                    f32(ctx, tensors_.at(name + ".bias")));
}

BoundaryFeatures BoundaryHead::build(ggml_context * ctx, ggml_tensor * encoded, ggml_tensor * words,
                                     ggml_tensor * labels, ggml_tensor * attention_mask) const {
    auto * text = ggml_get_rows(ctx, encoded, words);
    auto * queries = ggml_get_rows(ctx, encoded, labels);
    auto * left = ggml_concat(ctx, f32(ctx, tensors_.at("boundary_encoder.bos_state")), text, 1);
    auto * right = ggml_concat(ctx, text, f32(ctx, tensors_.at("boundary_encoder.eos_state")), 1);
    auto * joined = ggml_concat(ctx, linear(ctx, left, "boundary_encoder.left_projection"),
                               linear(ctx, right, "boundary_encoder.right_projection"), 0);
    auto * states = norm(ctx, linear(ctx, joined, "boundary_encoder.output_projection"), "boundary_encoder.layer_norm");
    const int64_t n = states->ne[1], d = dim / heads;
    auto split = [&](ggml_tensor * x) {
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x, d, heads, n), 0, 2, 1, 3));
    };
    for (int i = 0; i < attention_layers; ++i) {
        const auto p = "boundary_encoder.attention_blocks." + std::to_string(i) + ".";
        auto * qkv = linear(ctx, norm(ctx, states, p + "norm"), p + "qkv_projection");
        auto * q = split(part(ctx, qkv, dim, 0));
        auto * k = split(part(ctx, qkv, dim, 1));
        auto * v = split(part(ctx, qkv, dim, 2));
        auto * weights = ggml_soft_max(ctx, ggml_add(ctx, ggml_scale(ctx, mm(ctx, k, q), 1.f / std::sqrt(float(d))), attention_mask));
        auto * attended = mm(ctx, transpose(ctx, v), weights);
        attended = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), dim, n);
        states = ggml_add(ctx, states, linear(ctx, attended, p + "output_projection"));
    }
    for (int i = 0; i < refinement_layers; ++i) {
        const auto p = "boundary_encoder.refinement_blocks." + std::to_string(i) + ".";
        auto * value_gate = linear(ctx, norm(ctx, states, p + "norm"), p + "input_projection");
        auto * update = ggml_mul(ctx, part(ctx, value_gate, ffn_size, 0), ggml_silu(ctx, part(ctx, value_gate, ffn_size, 1)));
        states = ggml_add(ctx, states, linear(ctx, update, p + "output_projection"));
    }
    auto marginal = [&](const std::string & side) {
        return ggml_scale(ctx, mm(ctx, linear(ctx, states, "boundary_query_head." + side + "_boundary_projection"),
                                  linear(ctx, queries, "boundary_query_head." + side + "_query_projection")),
                          1.f / std::sqrt(float(dim)));
    };
    auto * inside = ggml_scale(ctx, mm(ctx, linear(ctx, text, "boundary_query_head.inside_text_projection"),
                                       linear(ctx, queries, "boundary_query_head.inside_query_projection")),
                                1.f / std::sqrt(float(dim)));
    return {states, queries, linear(ctx, text, "shared_pool_scorer.content_pooler.value_projection"),
            marginal("start"), marginal("end"), inside, linear(ctx, queries, "null_projection"),
            linear(ctx, states, "shared_pool_builder.start_projection"),
            linear(ctx, states, "shared_pool_builder.end_projection")};
}

ggml_tensor * BoundaryHead::score(ggml_context * ctx, const BoundaryFeatures & f, ggml_tensor * starts,
                                 ggml_tensor * ends, ggml_tensor * lengths, ggml_tensor * length_features,
                                 ggml_tensor *& compatibility) const {
    auto * ps = ggml_get_rows(ctx, f.pool_start, starts);
    auto * pe = ggml_get_rows(ctx, f.pool_end, ends);
    compatibility = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, ps, pe)), 1.f / std::sqrt(float(dim)));
    const std::string p = "shared_pool_scorer.";
    auto * start_rep = ggml_get_rows(ctx, linear(ctx, f.states, p + "start_projection"), starts);
    auto * end_rep = ggml_get_rows(ctx, linear(ctx, f.states, p + "end_projection"), ends);
    auto * candidate = ggml_add(ctx, ggml_add(ctx, ggml_add(ctx, start_rep, end_rep),
        linear(ctx, length_features, p + "length_projection")), linear(ctx, compatibility, p + "prior_projection"));
    auto * content_prefix = transpose(ctx, prefix(ctx, transpose(ctx, f.content)));
    auto * content = ggml_div(ctx, ggml_sub(ctx, ggml_get_rows(ctx, content_prefix, ends),
                                          ggml_get_rows(ctx, content_prefix, starts)), lengths);
    candidate = ggml_add(ctx, candidate, linear(ctx, norm(ctx, content, p + "content_pooler.layer_norm"), p + "content_projection"));
    candidate = norm(ctx, candidate, p + "candidate_norm");
    auto * query = linear(ctx, f.queries, p + "query_projection");
    auto * score = ggml_scale(ctx, mm(ctx, candidate, query), 1.f / std::sqrt(float(pair_dim)));
    const int64_t c = candidate->ne[1], q = query->ne[1];
    auto * film = linear(ctx, query, p + "film");
    auto * gamma = ggml_reshape_3d(ctx, part(ctx, film, pair_dim, 0), pair_dim, 1, q);
    auto * beta = ggml_reshape_3d(ctx, part(ctx, film, pair_dim, 1), pair_dim, 1, q);
    auto * conditioned = ggml_add(ctx, ggml_mul(ctx, ggml_repeat_4d(ctx, candidate, pair_dim, c, q, 1),
        ggml_add(ctx, ggml_fill(ctx, gamma, 1.f), gamma)), beta);
    auto * film_score = linear(ctx, ggml_gelu_erf(ctx, linear(ctx, conditioned, p + "film_output.0")), p + "film_output.3");
    score = ggml_add(ctx, score, ggml_reshape_2d(ctx, film_score, c, q));
    score = ggml_add(ctx, score, gather_columns(ctx, f.start, starts));
    score = ggml_add(ctx, score, gather_columns(ctx, f.end, ends));
    auto * mean = ggml_mean(ctx, f.inside);
    auto * inside_prefix = prefix(ctx, ggml_sub(ctx, f.inside, mean));
    auto * interval = ggml_sub(ctx, gather_columns(ctx, inside_prefix, ends), gather_columns(ctx, inside_prefix, starts));
    auto * row_lengths = ggml_reshape_2d(ctx, lengths, c, 1);
    auto * mean_rows = ggml_repeat_4d(ctx, mean, c, q, 1, 1);
    interval = ggml_add(ctx, interval, ggml_mul(ctx, mean_rows, row_lengths));
    return ggml_add(ctx, score, ggml_div(ctx, interval, ggml_sqrt(ctx, row_lengths)));
}
BoundaryRecordHead::BoundaryRecordHead(const GgufFile & file, ggml_context * weights) {
    if (file.string("gliner.boundary.record_variant") != "instances-v1") throw std::runtime_error("Unsupported boundary record variant");
    hidden = file.u32("deberta.hidden_size");
    dim = file.u32("gliner.boundary.record_dim");
    instances = file.u32("gliner.boundary.record_instance_queries");
    temperature = file.f32("gliner.boundary.record_temperature");
    if (dim < 1 || dim > 4096 || instances < 1 || instances > 256 || temperature <= 0) {
        throw std::runtime_error("Invalid boundary record dimensions or temperature");
    }
    const int boundary_dim = file.u32("gliner.boundary.boundary_dim");
    auto add = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        tensors_.emplace(name, file.tensor(weights, name, shape));
    };
    auto linear = [&](const std::string & name, int in, int out) {
        add(name + ".weight", {in, out});
        add(name + ".bias", {out});
    };
    linear("boundary_head.candidate_encoder", 2 * boundary_dim, hidden);
    for (const std::string name : {"inst_proj", "field_proj", "cand_proj", "q_proj", "k_proj"}) {
        linear("record_decoder." + name, hidden, dim);
    }
    linear("record_decoder.v_proj", hidden, hidden);
    linear("record_decoder.object_head", hidden, 1);
    linear("record_decoder.latent_seed_head", hidden, 1);
    add("record_decoder.null_embed", {dim});
    add("record_decoder.instance_embed", {hidden, instances});
}

ggml_tensor * BoundaryRecordHead::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, mm(ctx, tensors_.at(name + ".weight"), x), f32(ctx, tensors_.at(name + ".bias")));
}

ggml_tensor * BoundaryRecordHead::build(ggml_context * ctx, const BoundaryFeatures & f, ggml_tensor * starts,
                                       ggml_tensor * ends, gliner_record_mode mode, ggml_tensor *& objects) const {
    const int64_t count = starts ? starts->ne[0] : 0, fields = f.queries->ne[1];
    ggml_tensor * candidates = nullptr;
    if (count) {
        auto * endpoints = ggml_concat(ctx, ggml_get_rows(ctx, f.states, starts), ggml_get_rows(ctx, f.states, ends), 0);
        candidates = linear(ctx, endpoints, "boundary_head.candidate_encoder");
    }
    const std::string p = "record_decoder.";
    ggml_tensor * inst = nullptr;
    if (mode == GLINER_RECORD_LATENT) {
        if (!count) { objects = nullptr; return nullptr; }
        inst = ggml_reshape_2d(ctx, ggml_repeat_4d(ctx, candidates, hidden, count, fields, 1), hidden, count * fields);
        objects = linear(ctx, inst, p + "latent_seed_head");
    } else {
        inst = f32(ctx, tensors_.at(p + "instance_embed"));
        if (count) {
            auto * context = ggml_reshape_2d(ctx, ggml_repeat_4d(ctx, candidates, hidden, count, fields, 1), hidden, count * fields);
            auto * q = linear(ctx, inst, p + "q_proj");
            auto * k = linear(ctx, context, p + "k_proj");
            auto * v = linear(ctx, context, p + "v_proj");
            auto * attention = ggml_soft_max(ctx, ggml_scale(ctx, mm(ctx, k, q), 1.f / std::sqrt(float(dim))));
            inst = ggml_add(ctx, inst, mm(ctx, transpose(ctx, v), attention));
        }
        objects = linear(ctx, inst, p + "object_head");
    }
    auto * inst_q = linear(ctx, inst, p + "inst_proj");
    auto * field_q = ggml_reshape_3d(ctx, linear(ctx, f.queries, p + "field_proj"), dim, 1, fields);
    auto * query = ggml_add(ctx, ggml_repeat_4d(ctx, inst_q, dim, inst->ne[1], fields, 1), field_q);
    auto * keys = f32(ctx, tensors_.at(p + "null_embed"));
    if (count) keys = ggml_concat(ctx, keys, linear(ctx, candidates, p + "cand_proj"), 1);
    return mm(ctx, keys, query);
}
} // namespace gliner
