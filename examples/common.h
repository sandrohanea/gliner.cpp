#pragma once

#include "gliner/gliner.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gliner_examples {

inline std::vector<std::string> split_labels(const std::string & value) {
    if (value.empty() || value.back() == ',') throw std::runtime_error("Empty label");
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string label;
    while (std::getline(stream, label, ',')) {
        if (label.empty()) throw std::runtime_error("Empty label");
        result.push_back(label);
    }
    return result;
}

inline std::string escape_json(const std::string & value) {
    std::string result;
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c < 0x20) {
            const char * hex = "0123456789abcdef";
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else result += char(c);
    }
    return result;
}

inline int integer(const std::string & value) {
    size_t used;
    const int n = std::stoi(value, &used);
    if (used != value.size() || n < 0) throw std::invalid_argument("Expected a nonnegative integer");
    return n;
}

inline double number(const std::string & value) {
    size_t used;
    const double n = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(n)) throw std::invalid_argument("Expected a finite number");
    return n;
}

inline std::string read_text_file(const std::string & path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open text file");
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad()) throw std::runtime_error("Cannot read text file");
    return text;
}

struct TaskOptions {
    std::string name, prompt, activation;
    std::vector<std::string> labels, description_args;
    std::vector<std::optional<std::string>> descriptions;
    std::vector<const char *> label_ptrs, description_ptrs;
    bool multi_label = false, has_threshold = false;
    bool comma_labels = false, repeated_labels = false;
    double temperature = 1.0, threshold = 0.5;

    void validate(bool text_mode) {
        if (comma_labels && repeated_labels) throw std::invalid_argument("Use either --label or --labels within each task");
        if (labels.empty()) throw std::invalid_argument("Every task requires labels");
        if (text_mode && name.empty()) throw std::invalid_argument("--task is required for text inference");
        if (temperature <= 0 || threshold < 0 || threshold > 1 || (has_threshold && !multi_label)) {
            throw std::invalid_argument("Invalid task temperature or threshold");
        }
        if (activation.empty()) activation = text_mode && !multi_label ? "softmax" : "sigmoid";
        if (activation != "softmax" && activation != "sigmoid") throw std::invalid_argument("Unknown activation");
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i].empty() || std::find(labels.begin(), labels.begin() + i, labels[i]) != labels.begin() + i) {
                throw std::invalid_argument("Labels must be nonempty and unique within each task");
            }
        }
        descriptions.resize(labels.size());
        for (const auto & desc : description_args) {
            const size_t equal = desc.find('=');
            const auto found = std::find(labels.begin(), labels.end(), desc.substr(0, equal));
            if (equal == std::string::npos || found == labels.end()) throw std::invalid_argument("Expected --description LABEL=TEXT for the current task");
            const size_t index = static_cast<size_t>(found - labels.begin());
            if (descriptions[index]) throw std::invalid_argument("Duplicate label description");
            descriptions[index] = desc.substr(equal + 1);
        }
    }

    gliner_classification_task input() {
        label_ptrs.clear();
        description_ptrs.clear();
        for (size_t i = 0; i < labels.size(); ++i) {
            label_ptrs.push_back(labels[i].c_str());
            description_ptrs.push_back(descriptions[i] ? descriptions[i]->c_str() : nullptr);
        }
        return {name.c_str(), label_ptrs.data(), static_cast<int>(labels.size()), prompt.c_str(), description_ptrs.data()};
    }
};

template <typename T>
void array(std::ostream & out, const T * values, size_t size) {
    out << '[';
    for (size_t i = 0; i < size; ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

} // namespace gliner_examples
