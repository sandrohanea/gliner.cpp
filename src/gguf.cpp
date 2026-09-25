#include "internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gliner {

ModelReader::ModelReader(const void * data, size_t size)
    : data_(static_cast<const unsigned char *>(data)), size_(size) {}

ModelReader::ModelReader(const gliner_model_loader & loader, uint64_t size)
    : loader_(loader), size_(size) {}

void ModelReader::read_exact(void * output, size_t count) {
    if (position_ > size_ || count > size_ - position_) throw std::runtime_error("Truncated GGUF input");
    auto * bytes = static_cast<unsigned char *>(output);
    while (count) {
        const size_t requested = std::min(count, size_t(8 * 1024 * 1024));
        const size_t n = loader_.read(loader_.context, bytes, requested);
        if (n > requested) throw std::runtime_error("Model loader returned more bytes than requested");
        if (n == 0) {
            throw std::runtime_error(loader_.eof(loader_.context) ?
                "Unexpected end of GGUF stream" : "Model loader read failed (no progress before EOF)");
        }
        position_ += n;
        bytes += n;
        count -= n;
    }
}

void ModelReader::read_at(void * output, uint64_t offset, size_t count) {
    if (offset > size_ || count > size_ - offset) throw std::runtime_error("Truncated GGUF input");
    if (data_) {
        if (count) std::memcpy(output, data_ + static_cast<size_t>(offset), count);
        return;
    }
    if (offset < position_) throw std::runtime_error("GGUF parser requested a nonsequential stream read");
    if (offset > position_) {
        std::array<unsigned char, 64 * 1024> discard;
        while (position_ < offset) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(discard.size(), offset - position_));
            read_exact(discard.data(), n);
        }
    }
    read_exact(output, count);
}

size_t ModelReader::gguf_read(void * context, void * output, uint64_t offset, size_t count) noexcept {
    auto & reader = *static_cast<ModelReader *>(context);
    if (reader.error_) return 0;
    try {
        reader.read_at(output, offset, count);
        return count;
    } catch (...) {
        reader.error_ = std::current_exception();
        return 0;
    }
}

void ModelReader::rethrow_error() const {
    if (error_) std::rethrow_exception(error_);
}

GgufFile::GgufFile(ModelReader & reader) {
    gguf_init_params params = {};
    params.no_alloc = true;
    ctx_ = gguf_init_from_callback(ModelReader::gguf_read, &reader, 8 * 1024 * 1024, reader.size(), params);
    try {
        reader.rethrow_error();
        if (!ctx_) throw std::runtime_error("Cannot read GGUF metadata (invalid or truncated input)");
        const auto family = string("general.architecture");
        if (family != "gliner2.5-decide" && family != "gliner2") {
            throw std::runtime_error("GGUF is not a supported GLiNER2 checkpoint");
        }
        if (has("gliner.tensor_name_map.original") || has("gliner.tensor_name_map.stored")) {
            const auto original = strings("gliner.tensor_name_map.original");
            const auto stored = strings("gliner.tensor_name_map.stored");
            if (original.size() != stored.size()) throw std::runtime_error("Invalid tensor alias map");
            std::unordered_map<std::string, bool> used;
            for (size_t i = 0; i < original.size(); ++i) {
                if (original[i].empty() || stored[i].empty() || !aliases_.emplace(original[i], stored[i]).second ||
                    !used.emplace(stored[i], true).second || gguf_find_tensor(ctx_, stored[i].c_str()) < 0) {
                    throw std::runtime_error("Invalid tensor alias mapping");
                }
            }
        }
        const uint64_t start = gguf_get_data_offset(ctx_);
        for (int i = 0; i < tensor_count(); ++i) {
            const uint64_t offset = gguf_get_tensor_offset(ctx_, i);
            const uint64_t size = gguf_get_tensor_size(ctx_, i);
            if (start > reader.size() || offset > reader.size() - start || size > reader.size() - start - offset) {
                throw std::runtime_error("Truncated GGUF tensor payload");
            }
        }
    } catch (...) {
        if (ctx_) gguf_free(ctx_);
        ctx_ = nullptr;
        throw;
    }
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

int GgufFile::tensor_count() const {
    const int64_t count = gguf_get_n_tensors(ctx_);
    if (count > std::numeric_limits<int>::max()) throw std::runtime_error("Too many GGUF tensors");
    return static_cast<int>(count);
}

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
    const auto alias = aliases_.find(name);
    const auto & stored = alias == aliases_.end() ? name : alias->second;
    const int64_t id = gguf_find_tensor(ctx_, stored.c_str());
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
    ggml_set_name(tensor, stored.c_str());
    const size_t bytes = gguf_get_tensor_size(ctx_, id);
    if (ggml_nbytes(tensor) != bytes) throw std::runtime_error("GGUF tensor size mismatch");
    return tensor;
}

void GgufFile::load_weights(ggml_context * ctx, ModelReader & reader) const {
    std::vector<unsigned char> data(8 * 1024 * 1024);
    for (int id = 0; id < tensor_count(); ++id) {
        auto * tensor = ggml_get_tensor(ctx, gguf_get_tensor_name(ctx_, id));
        const uint64_t start = gguf_get_data_offset(ctx_) + gguf_get_tensor_offset(ctx_, id);
        const size_t bytes = gguf_get_tensor_size(ctx_, id);
        for (size_t offset = 0; offset < bytes;) {
            const size_t chunk = std::min(data.size(), bytes - offset);
            reader.read_at(data.data(), start + offset, chunk);
            if (tensor) ggml_backend_tensor_set(tensor, data.data(), offset, chunk);
            offset += chunk;
        }
    }
}

} // namespace gliner
