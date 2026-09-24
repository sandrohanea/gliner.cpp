#include "internal.h"
#include <stdexcept>

namespace gliner {

std::vector<std::string> classification_schema(
    const std::string & task, const std::vector<std::string> & labels,
    const std::string & prompt) {
    if (task.empty() || labels.empty()) throw std::invalid_argument("Task and labels are required");
    std::vector<std::string> result = {"(", "[P]", prompt.empty() ? task : task + ": " + prompt, "("};
    for (const auto & label : labels) {
        if (label.empty()) throw std::invalid_argument("Empty classification label");
        result.push_back("[L]");
        result.push_back(label);
    }
    result.push_back(")");
    result.push_back(")");
    return result;
}

} // namespace gliner
