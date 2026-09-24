#include "gliner/gliner.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::string> split_labels(const std::string & value) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string label;
    while (std::getline(stream, label, ',')) {
        if (label.empty()) throw std::runtime_error("Empty label");
        result.push_back(label);
    }
    return result;
}

std::string escape_json(const std::string & value) {
    std::string result;
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { result += '\\'; result += char(c); }
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else if (c < 0x20) {
            const char * hex = "0123456789abcdef";
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else result += char(c);
    }
    return result;
}

void usage() {
    std::cerr << "Usage: gliner-classify --model model.gguf --inspect\n"
                 "       gliner-classify --model model.gguf --labels a,b,c --states label_states.txt [--threads N]\n"
                 "Each nonempty states line contains hidden_size floats for one contextual [L] marker.\n";
}
}

int main(int argc, char ** argv) {
    try {
        std::string model_path, labels_string, states_path;
        int threads = 1;
        bool inspect = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--inspect") inspect = true;
            else if ((arg == "--model" || arg == "--labels" || arg == "--states" || arg == "--threads") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--model") model_path = value;
                else if (arg == "--labels") labels_string = value;
                else if (arg == "--states") states_path = value;
                else threads = std::stoi(value);
            } else { usage(); return 2; }
        }
        if (model_path.empty()) { usage(); return 2; }
        std::unique_ptr<gliner_context, decltype(&gliner_free)> model(
            gliner_init_from_file(model_path.c_str()), gliner_free);
        if (!model) throw std::runtime_error(gliner_last_error());
        if (inspect) {
            std::cout << "architecture: gliner2.5-decide\n"
                      << "tensors: " << gliner_model_n_tensors(model.get()) << '\n'
                      << "hidden_size: " << gliner_model_hidden_size(model.get()) << '\n'
                      << "encoder_layers: " << gliner_model_n_layers(model.get()) << '\n';
            return 0;
        }
        if (labels_string.empty() || states_path.empty()) { usage(); return 2; }
        const auto labels = split_labels(labels_string);
        std::ifstream file(states_path);
        if (!file) throw std::runtime_error("Cannot open states file");
        std::vector<float> states;
        std::string line;
        size_t count = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream row(line);
            float value;
            size_t width = 0;
            while (row >> value) { states.push_back(value); ++width; }
            if (!row.eof() || width != static_cast<size_t>(gliner_model_hidden_size(model.get()))) {
                throw std::runtime_error("Each states line must contain hidden_size floats");
            }
            ++count;
        }
        if (count != labels.size()) throw std::runtime_error("Labels and states rows differ");
        std::unique_ptr<gliner_state, decltype(&gliner_free_state)> state(
            gliner_init_state(model.get()), gliner_free_state);
        if (!state) throw std::runtime_error(gliner_last_error());
        if (gliner_score_label_states(model.get(), state.get(), states.data(),
                                      static_cast<int>(count), threads) != GLINER_STATUS_OK) {
            throw std::runtime_error(gliner_last_error());
        }
        const gliner_score * scores = gliner_get_scores(state.get());
        const size_t winner = static_cast<size_t>(std::max_element(scores, scores + count,
            [](const gliner_score & a, const gliner_score & b) { return a.logit < b.logit; }) - scores);
        std::cout << std::setprecision(7) << "{\"label\":\"" << escape_json(labels[winner])
                  << "\",\"scores\":[";
        for (size_t i = 0; i < count; ++i) {
            if (i) std::cout << ',';
            std::cout << "{\"label\":\"" << escape_json(labels[i])
                      << "\",\"logit\":" << scores[i].logit
                      << ",\"probability\":" << scores[i].probability << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
