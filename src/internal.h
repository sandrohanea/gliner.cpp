#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gliner {

class GgufFile {
public:
    explicit GgufFile(const std::string & path);
    ~GgufFile();
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;
    int u32(const char * key) const;
    bool has(const char * key) const;
    float f32(const char * key) const;
    std::string string(const char * key) const;
    std::vector<std::string> strings(const char * key) const;
    std::vector<uint32_t> integers(const char * key) const;
    std::vector<double> doubles(const char * key) const;
    int tensor_count() const;
    ggml_tensor * tensor(ggml_context * ctx, const std::string & name,
                        std::initializer_list<int64_t> shape) const;
    void load_weights(ggml_context * ctx) const;
private:
    int64_t array(const char * key, gguf_type type) const;
    std::string path_;
    gguf_context * ctx_ = nullptr;
};

struct Tokenized {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;
};

class Tokenizer {
public:
    explicit Tokenizer(const GgufFile & file);
    Tokenized encode(const std::string & text, const std::string & task,
                     const std::vector<std::string> & labels, const std::string & prompt,
                     const std::vector<std::optional<std::string>> & descriptions, int max_words, int max_tokens) const;
private:
    struct Node {
        std::unordered_map<unsigned char, size_t> children;
        int32_t id = -1;
    };
    std::vector<Node> trie_;
    std::vector<double> scores_;
    std::vector<std::string> added_;
    std::vector<uint32_t> added_ids_, ranges_;
    std::unordered_map<int32_t, std::string> lower_;
    int32_t unknown_;
    double unknown_score_;
    unsigned flags(int32_t cp) const;
    std::string lowercase(const std::vector<int32_t> & word) const;
    std::vector<std::string> words(const std::string & text, int max_words) const;
    std::string normalize(const std::string & text) const;
    void unigram(const std::string & text, std::vector<int32_t> & ids) const;
    void segment(const std::string & text, std::vector<int32_t> & ids) const;
};

class Deberta {
public:
    Deberta(const GgufFile & file, ggml_context * weights);
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * ids,
                       ggml_tensor * c2p_indices, ggml_tensor * p2c_indices,
                       std::vector<ggml_tensor *> & hidden_states) const;
    void relative_indices(int tokens, std::vector<int32_t> & c2p, std::vector<int32_t> & p2c) const;
    int hidden, heads, layers, intermediate, vocab, buckets, max_relative;
    float epsilon;
private:
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
};

void validate_deberta_metadata(const GgufFile & file);

} // namespace gliner
