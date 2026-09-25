#pragma once

#include "gliner/gliner.h"
#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gliner {

class ModelReader {
public:
    ModelReader(const void * data, size_t size);
    explicit ModelReader(const gliner_model_loader & loader, uint64_t size = UINT64_MAX);
    uint64_t size() const { return size_; }
    void read_at(void * output, uint64_t offset, size_t count);
    void rethrow_error() const;
    static size_t gguf_read(void * context, void * output, uint64_t offset, size_t count) noexcept;
private:
    void read_exact(void * output, size_t count);
    const unsigned char * data_ = nullptr;
    gliner_model_loader loader_ = {};
    uint64_t size_;
    uint64_t position_ = 0;
    std::exception_ptr error_;
};

class GgufFile {
public:
    explicit GgufFile(ModelReader & reader);
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
    void load_weights(ggml_context * ctx, ModelReader & reader) const;
private:
    int64_t array(const char * key, gguf_type type) const;
    gguf_context * ctx_ = nullptr;
};

struct Word {
    std::string text;
    size_t start;
    size_t end;
};

struct Tokenized {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;
    std::vector<gliner_task_result> tasks;
    std::vector<Word> words;
    std::vector<int32_t> word_positions;
    int32_t prompt_position = 0;
};

struct ClassificationTask {
    std::string name;
    std::vector<std::string> labels;
    std::string prompt;
    std::vector<std::optional<std::string>> descriptions;
};

class Tokenizer {
public:
    explicit Tokenizer(const GgufFile & file);
    Tokenized encode(const std::string & text, const std::vector<ClassificationTask> & tasks,
                     int max_words, int max_tokens, bool entities = false) const;
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
    std::vector<Word> words(const std::string & text, int max_words) const;
    std::string normalize(const std::string & text) const;
    void unigram(const std::string & text, std::vector<int32_t> & ids) const;
    void segment(const std::string & text, std::vector<int32_t> & ids) const;
};

class SpanHead {
public:
    SpanHead(const GgufFile & file, ggml_context * weights);
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * encoded, ggml_tensor * words,
                       ggml_tensor * labels, ggml_tensor * prompt, ggml_tensor * starts,
                       ggml_tensor * ends, ggml_tensor * count_index, ggml_tensor *& count_logits) const;
    int hidden;
    int max_width;
private:
    std::unordered_map<std::string, ggml_tensor *> tensors_;
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * projection(ggml_context * ctx, ggml_tensor * x, const std::string & name,
                            const char * final_layer) const;
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
