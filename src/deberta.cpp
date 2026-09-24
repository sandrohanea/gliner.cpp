#include "internal.h"
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

} // namespace gliner
