#include "internal.h"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace gliner {

GgufFile::GgufFile(const std::string & path) : path_(path) {
    gguf_init_params params = {};
    params.no_alloc = true;
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) throw std::runtime_error("Cannot open GGUF: " + path);
    try {
        if (string("general.architecture") == "gliner2.5-decide") return;
    } catch (...) {
        gguf_free(ctx_);
        ctx_ = nullptr;
        throw;
    }
    gguf_free(ctx_);
    ctx_ = nullptr;
    throw std::runtime_error("GGUF is not a GLiNER2.5-Decide checkpoint");
}

GgufFile::~GgufFile() { if (ctx_) gguf_free(ctx_); }

int GgufFile::u32(const char * key) const {
    const int64_t index = gguf_find_key(ctx_, key);
    if (index < 0 || gguf_get_kv_type(ctx_, index) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("Missing GGUF u32 key: ") + key);
    }
    const uint32_t value = gguf_get_val_u32(ctx_, index);
    if (value > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("Oversized GGUF value: ") + key);
    }
    return static_cast<int>(value);
}

std::string GgufFile::string(const char * key) const {
    const int64_t index = gguf_find_key(ctx_, key);
    if (index < 0 || gguf_get_kv_type(ctx_, index) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("Missing GGUF string key: ") + key);
    }
    return gguf_get_val_str(ctx_, index);
}

int GgufFile::tensor_count() const { return static_cast<int>(gguf_get_n_tensors(ctx_)); }

ggml_tensor * GgufFile::read_tensor(ggml_context * ctx, const char * name,
                                    std::vector<unsigned char> & data) const {
    const int64_t id = gguf_find_tensor(ctx_, name);
    if (id < 0) throw std::runtime_error(std::string("Missing tensor: ") + name);
    const ggml_type type = gguf_get_tensor_type(ctx_, id);
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        throw std::runtime_error(std::string("Unsupported tensor type: ") + name);
    }
    const int64_t * dims = gguf_get_tensor_ne(ctx_, id);
    const std::string tensor_name(name);
    const int n_dims = tensor_name.size() >= 7 && tensor_name.substr(tensor_name.size() - 7) == ".weight" ? 2 : 1;
    ggml_tensor * tensor = ggml_new_tensor(ctx, type, n_dims, dims);
    if (!tensor) throw std::runtime_error("Failed to allocate tensor");
    ggml_set_name(tensor, name);
    const size_t bytes = gguf_get_tensor_size(ctx_, id);
    if (ggml_nbytes(tensor) != bytes) throw std::runtime_error("GGUF tensor size mismatch");
    std::ifstream input(path_, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot reopen GGUF file");
    input.seekg(static_cast<std::streamoff>(gguf_get_data_offset(ctx_) + gguf_get_tensor_offset(ctx_, id)));
    data.resize(bytes);
    input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error(std::string("Cannot read tensor: ") + name);
    return tensor;
}

} // namespace gliner
