#include "internal.h"

#include <fstream>
#include <cmath>
#include <cstring>
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

bool GgufFile::has(const char * key) const { return gguf_find_key(ctx_, key) >= 0; }

float GgufFile::f32(const char * key) const {
    const int64_t id = gguf_find_key(ctx_, key);
    if (id < 0 || gguf_get_kv_type(ctx_, id) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(std::string("Missing GGUF f32 key: ") + key);
    }
    const float value = gguf_get_val_f32(ctx_, id);
    if (!std::isfinite(value)) throw std::runtime_error(std::string("Nonfinite GGUF value: ") + key);
    return value;
}

int64_t GgufFile::array(const char * key, gguf_type type) const {
    const int64_t id = gguf_find_key(ctx_, key);
    if (id < 0 || gguf_get_kv_type(ctx_, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_, id) != type) {
        throw std::runtime_error(std::string("Missing or invalid GGUF array: ") + key);
    }
    return id;
}

std::vector<std::string> GgufFile::strings(const char * key) const {
    const int64_t id = array(key, GGUF_TYPE_STRING);
    std::vector<std::string> result;
    for (size_t i = 0; i < gguf_get_arr_n(ctx_, id); ++i) result.emplace_back(gguf_get_arr_str(ctx_, id, i));
    return result;
}

std::vector<uint32_t> GgufFile::integers(const char * key) const {
    const int64_t id = array(key, GGUF_TYPE_UINT32);
    std::vector<uint32_t> result(gguf_get_arr_n(ctx_, id));
    if (!result.empty()) std::memcpy(result.data(), gguf_get_arr_data(ctx_, id), result.size() * sizeof(uint32_t));
    return result;
}

std::vector<double> GgufFile::doubles(const char * key) const {
    const int64_t id = array(key, GGUF_TYPE_FLOAT64);
    std::vector<double> result(gguf_get_arr_n(ctx_, id));
    if (!result.empty()) std::memcpy(result.data(), gguf_get_arr_data(ctx_, id), result.size() * sizeof(double));
    return result;
}

ggml_tensor * GgufFile::tensor(ggml_context * ctx, const std::string & name,
                             std::initializer_list<int64_t> shape) const {
    const int64_t id = gguf_find_tensor(ctx_, name.c_str());
    if (id < 0) throw std::runtime_error("Missing tensor: " + name);
    const ggml_type type = gguf_get_tensor_type(ctx_, id);
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        throw std::runtime_error("Unsupported tensor type: " + name);
    }
    const int64_t * dims = gguf_get_tensor_ne(ctx_, id);
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        const int64_t expected = i < shape.size() ? shape.begin()[i] : 1;
        if (dims[i] != expected || expected <= 0) throw std::runtime_error("Invalid tensor shape: " + name);
    }
    ggml_tensor * tensor = ggml_new_tensor(ctx, type, static_cast<int>(shape.size()), dims);
    if (!tensor) throw std::runtime_error("Failed to allocate tensor");
    ggml_set_name(tensor, name.c_str());
    const size_t bytes = gguf_get_tensor_size(ctx_, id);
    if (ggml_nbytes(tensor) != bytes) throw std::runtime_error("GGUF tensor size mismatch");
    return tensor;
}

void GgufFile::load_weights(ggml_context * ctx) const {
    std::ifstream input(path_, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot reopen GGUF file");
    std::vector<unsigned char> data(8 * 1024 * 1024);
    for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        const int64_t id = gguf_find_tensor(ctx_, tensor->name);
        input.seekg(static_cast<std::streamoff>(gguf_get_data_offset(ctx_) + gguf_get_tensor_offset(ctx_, id)));
        const size_t bytes = ggml_nbytes(tensor);
        for (size_t offset = 0; offset < bytes;) {
            const size_t chunk = std::min(data.size(), bytes - offset);
            input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(chunk));
            if (!input) throw std::runtime_error(std::string("Cannot read tensor: ") + tensor->name);
            ggml_backend_tensor_set(tensor, data.data(), offset, chunk);
            offset += chunk;
        }
    }
}

} // namespace gliner
