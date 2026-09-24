#pragma once

#include <ggml.h>
#include <gguf.h>
#include <string>
#include <vector>

namespace gliner {

class GgufFile {
public:
    explicit GgufFile(const std::string & path);
    ~GgufFile();
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;
    int u32(const char * key) const;
    std::string string(const char * key) const;
    int tensor_count() const;
    ggml_tensor * read_tensor(ggml_context * ctx, const char * name,
                              std::vector<unsigned char> & bytes) const;
private:
    std::string path_;
    gguf_context * ctx_ = nullptr;
};

void validate_deberta_metadata(const GgufFile & file);
std::vector<std::string> classification_schema(
    const std::string & task, const std::vector<std::string> & labels,
    const std::string & prompt = "");

} // namespace gliner
