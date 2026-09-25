#include "gliner/gliner.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

std::vector<std::string> split_labels(const std::string & value) {
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

std::string escape_json(const std::string & value) {
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

int integer(const std::string & value) {
    size_t used;
    const int n = std::stoi(value, &used);
    if (used != value.size() || n < 0) throw std::invalid_argument("Expected a nonnegative integer");
    return n;
}

double number(const std::string & value) {
    size_t used;
    const double n = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(n)) throw std::invalid_argument("Expected a finite number");
    return n;
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

void write_result(std::ostream & out, const TaskOptions & task, const gliner_score * scores) {
    const size_t count = task.labels.size();
    const size_t winner = static_cast<size_t>(std::max_element(scores, scores + count,
        [](const gliner_score & a, const gliner_score & b) { return a.logit < b.logit; }) - scores);
    std::vector<double> probabilities(count);
    double total = 0;
    for (size_t i = 0; i < count; ++i) {
        if (task.activation == "softmax") {
            probabilities[i] = std::exp((static_cast<double>(scores[i].logit) - scores[winner].logit) / task.temperature);
            total += probabilities[i];
        } else {
            const double x = scores[i].logit / task.temperature;
            probabilities[i] = x >= 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
        }
    }
    if (task.activation == "softmax") for (auto & p : probabilities) p /= total;
    if (task.multi_label) {
        out << "\"labels\":[";
        bool first = true;
        for (size_t i = 0; i < count; ++i) {
            if (probabilities[i] < task.threshold) continue;
            if (!first) out << ',';
            out << '"' << escape_json(task.labels[i]) << '"';
            first = false;
        }
        out << ']';
    } else out << "\"label\":\"" << escape_json(task.labels[winner]) << '"';
    out << ",\"scores\":[";
    for (size_t i = 0; i < count; ++i) {
        if (i) out << ',';
        out << "{\"label\":\"" << escape_json(task.labels[i])
            << "\",\"logit\":" << scores[i].logit << ",\"probability\":" << probabilities[i] << '}';
    }
    out << ']';
}

void usage() {
    std::cout <<
        "Usage: gliner-classify --build-info\n"
        "       gliner-classify --model model.gguf --inspect\n"
        "       gliner-classify --model model.gguf --task intent --labels a,b,c --text \"text\"\n"
        "       gliner-classify --model model.gguf --text \"text\" --task intent --labels a,b --task sentiment --labels positive,negative\n"
        "       gliner-classify --model model.gguf --labels a,b,c --states label_states.txt\n"
        "Options:\n"
        "  Backend is selected at build time using GGML_CUDA/GGML_METAL, not by runtime flags.\n"
        "  --text-file FILE        UTF-8 input instead of --text\n"
        "  --task NAME             Start a task group; repeated tasks share one text and one encoder pass\n"
        "  --label LABEL           Repeat instead of comma-separated --labels, within the current task\n"
        "  --prompt TEXT           Current task's prompt\n"
        "  --description LABEL=TEXT  Repeat for label descriptions (encoded in label order)\n"
        "  --threads N             CPU threads (default 1; ignored on GPUs)\n"
        "  --max-words N           Truncate text after N upstream word tokens (0 = unlimited)\n"
        "  --max-tokens N          Hard limit over ALL task schemas + text, 1..4096 (default 512)\n"
        "  --multi-label           Current task returns labels meeting --threshold (default 0.5)\n"
        "  --threshold T           Current task's multi-label probability cutoff in [0,1]\n"
        "  --activation MODE       Current task: sigmoid or softmax (default softmax for text, sigmoid for states/multi-label)\n"
        "  --temperature T         Current task's positive temperature (default 1; logits unchanged)\n"
        "  Task options reset at each --task. Joint scores may differ from separate calls.\n"
        "  --debug                 Include token IDs, label positions and contextual states in JSON\n"
        "  --dump-hidden FILE      Diagnostic LE float32 layer outputs, preceded by u32 tokens, u32 hidden\n";
}

void write_u32(std::ostream & out, uint32_t value) {
    char bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = static_cast<char>(value >> (i * 8));
    out.write(bytes, 4);
}

void dump_layer(int layer, const float * values, int tokens, int hidden, void * user_data) {
    auto & out = *static_cast<std::ofstream *>(user_data);
    if (layer == 0) { write_u32(out, static_cast<uint32_t>(tokens)); write_u32(out, static_cast<uint32_t>(hidden)); }
    for (size_t i = 0; i < static_cast<size_t>(tokens) * hidden; ++i) {
        uint32_t bits;
        std::memcpy(&bits, values + i, sizeof(bits));
        write_u32(out, bits);
    }
    if (!out) throw std::runtime_error("Cannot write hidden-state dump");
}

template <typename T>
void array(std::ostream & out, const T * values, size_t size) {
    out << '[';
    for (size_t i = 0; i < size; ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

int run(const std::vector<std::string> & args) {
    try {
        std::string model_path, states_path, text, text_file, dump_path;
        std::vector<TaskOptions> tasks(1);
        auto params = gliner_default_batch_params();
        bool inspect = false, has_text = false, debug = false;
        bool has_task = false, text_options = false;
        bool build_info = false;
        for (size_t i = 1; i < args.size(); ++i) {
            const auto & arg = args[i];
            auto & task = tasks.back();
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg == "--inspect") { inspect = true; continue; }
            if (arg == "--build-info") { build_info = true; continue; }
            if (arg == "--backend" || arg == "--device" || arg == "--list-devices") {
                throw std::invalid_argument("Backend selection is build-time only; configure GGML_CUDA/GGML_METAL with CMake");
            }
            if (arg == "--multi-label") { task.multi_label = true; continue; }
            if (arg == "--debug") { debug = true; continue; }
            if (i + 1 == args.size()) throw std::invalid_argument("Missing value for " + arg);
            const auto & value = args[++i];
            if (arg == "--model") model_path = value;
            else if (arg == "--states") states_path = value;
            else if (arg == "--text") { text = value; has_text = true; }
            else if (arg == "--text-file") text_file = value;
            else if (arg == "--task") {
                if (has_task) tasks.emplace_back();
                tasks.back().name = value;
                has_task = true;
            }
            else if (arg == "--labels") { task.labels = split_labels(value); task.comma_labels = true; }
            else if (arg == "--label") { task.labels.push_back(value); task.repeated_labels = true; }
            else if (arg == "--prompt") { task.prompt = value; text_options = true; }
            else if (arg == "--description") { task.description_args.push_back(value); text_options = true; }
            else if (arg == "--threads") params.n_threads = integer(value);
            else if (arg == "--max-words") { params.max_words = integer(value); text_options = true; }
            else if (arg == "--max-tokens") { params.max_tokens = integer(value); text_options = true; }
            else if (arg == "--temperature") task.temperature = number(value);
            else if (arg == "--threshold") { task.threshold = number(value); task.has_threshold = true; }
            else if (arg == "--activation") task.activation = value;
            else if (arg == "--dump-hidden") dump_path = value;
            else throw std::invalid_argument("Unknown option: " + arg);
        }
        if (build_info) {
            if (args.size() != 2) throw std::invalid_argument("--build-info must be used alone");
            std::cout << "{\"backend\":\"" << gliner_build_backend() << "\"}\n";
            return 0;
        }
        if (model_path.empty()) throw std::invalid_argument("--model is required");
        if (has_text && !text_file.empty()) throw std::invalid_argument("Use either --text or --text-file");
        const bool text_mode = has_text || !text_file.empty();
        if (!inspect && (text_mode == !states_path.empty())) {
            throw std::invalid_argument("Provide labels and exactly one of --text, --text-file or --states");
        }
        if (!text_mode && (text_options || has_task || debug || !dump_path.empty())) {
            throw std::invalid_argument("Text options require --text or --text-file");
        }
        if (params.n_threads <= 0 || params.max_tokens <= 0 || params.max_tokens > 4096) {
            throw std::invalid_argument("Invalid thread count or token limit");
        }
        size_t count = 0;
        if (!inspect) {
            for (size_t t = 0; t < tasks.size(); ++t) {
                tasks[t].validate(text_mode);
                for (size_t previous = 0; previous < t; ++previous) {
                    if (tasks[previous].name == tasks[t].name) throw std::invalid_argument("Task names must be unique");
                }
                if (tasks[t].labels.size() > static_cast<size_t>(std::numeric_limits<int>::max()) - count) {
                    throw std::invalid_argument("Too many labels");
                }
                count += tasks[t].labels.size();
            }
            if (text_mode && count > static_cast<size_t>(params.max_tokens)) {
                throw std::invalid_argument("Total labels across all tasks must fit max_tokens");
            }
        }
        std::unique_ptr<gliner_context, decltype(&gliner_free)> model(
            gliner_init_from_file(model_path.c_str()), gliner_free);
        if (!model) throw std::runtime_error(gliner_last_error());
        if (inspect) {
            std::cout << "architecture: gliner2.5-decide\n"
                      << "tensors: " << gliner_model_n_tensors(model.get()) << '\n'
                      << "hidden_size: " << gliner_model_hidden_size(model.get()) << '\n'
                      << "encoder_layers: " << gliner_model_n_layers(model.get()) << '\n'
                      << "backend: " << gliner_model_backend_name(model.get()) << '\n'
                      << "device: " << gliner_model_device_name(model.get()) << '\n'
                      << "text_inference: " << (gliner_model_supports_text(model.get()) ? "yes" : "no (legacy GGUF)") << '\n';
            return 0;
        }
        std::unique_ptr<gliner_state, decltype(&gliner_free_state)> state(
            gliner_init_state(model.get()), gliner_free_state);
        if (!state) throw std::runtime_error(gliner_last_error());
        std::ofstream dump;
        if (!dump_path.empty()) {
            if (std::filesystem::exists(std::filesystem::u8path(dump_path))) throw std::invalid_argument("Hidden-state dump already exists");
            dump.open(std::filesystem::u8path(dump_path), std::ios::binary);
            if (!dump) throw std::runtime_error("Cannot open hidden-state dump");
            gliner_set_eval_callback(state.get(), dump_layer, &dump);
        }
        int status;
        if (text_mode) {
            if (!text_file.empty()) {
                std::ifstream input(std::filesystem::u8path(text_file), std::ios::binary);
                if (!input) throw std::runtime_error("Cannot open text file");
                text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
                if (input.bad()) throw std::runtime_error("Cannot read text file");
            }
            if (text.find('\0') != std::string::npos) throw std::invalid_argument("Text contains NUL");
            std::vector<gliner_classification_task> batch;
            for (auto & task : tasks) batch.push_back(task.input());
            status = gliner_classify_text_batch(model.get(), state.get(), text.c_str(), batch.data(),
                                                static_cast<int>(batch.size()), &params);
        } else {
            std::ifstream file(std::filesystem::u8path(states_path));
            if (!file) throw std::runtime_error("Cannot open states file");
            std::vector<float> states;
            std::string line;
            size_t rows = 0;
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                std::istringstream row(line);
                float value;
                size_t width = 0;
                while (row >> value) { states.push_back(value); ++width; }
                if (!row.eof() || width != static_cast<size_t>(gliner_model_hidden_size(model.get()))) {
                    throw std::runtime_error("Each states line must contain hidden_size floats");
                }
                ++rows;
            }
            if (file.bad() || rows != count) throw std::runtime_error("Labels and states rows differ or states file is unreadable");
            status = gliner_score_label_states(model.get(), state.get(), states.data(), static_cast<int>(count), params.n_threads);
        }
        if (status != GLINER_STATUS_OK) throw std::runtime_error(gliner_last_error());
        if (dump.is_open()) { dump.close(); if (!dump) throw std::runtime_error("Cannot finish hidden-state dump"); }
        const gliner_score * scores = gliner_get_scores(state.get());
        const auto * ranges = gliner_get_task_results(state.get());
        if (gliner_n_scores(state.get()) != static_cast<int>(count) ||
            (text_mode && gliner_n_tasks(state.get()) != static_cast<int>(tasks.size()))) {
            throw std::runtime_error("Unexpected classification result shape");
        }
        std::ostringstream output;
        output << std::setprecision(9) << '{';
        if (tasks.size() > 1) {
            output << "\"tasks\":[";
            for (size_t t = 0; t < tasks.size(); ++t) {
                if (ranges[t].n_scores != static_cast<int>(tasks[t].labels.size())) {
                    throw std::runtime_error("Task score count does not match labels");
                }
                if (t) output << ',';
                output << "{\"task\":\"" << escape_json(tasks[t].name) << "\",";
                write_result(output, tasks[t], scores + ranges[t].score_offset);
                output << '}';
            }
            output << ']';
        } else write_result(output, tasks[0], scores);
        if (debug) {
            output << ",\"backend\":\"" << gliner_model_backend_name(model.get())
                   << "\",\"device\":\"" << escape_json(gliner_model_device_name(model.get())) << '"'
                   << ",\"input_ids\":";
            array(output, gliner_get_token_ids(state.get()), static_cast<size_t>(gliner_n_tokens(state.get())));
            output << ",\"label_positions\":";
            array(output, gliner_get_label_positions(state.get()), count);
            output << ",\"label_states\":";
            array(output, gliner_get_label_states(state.get()), count * gliner_model_hidden_size(model.get()));
            std::vector<int> offsets;
            for (size_t t = 0; t < tasks.size(); ++t) offsets.push_back(ranges[t].score_offset);
            offsets.push_back(static_cast<int>(count));
            output << ",\"task_offsets\":";
            array(output, offsets.data(), offsets.size());
        }
        output << "}\n";
        std::cout << output.str();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t ** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0, nullptr, nullptr);
        if (!size) { std::cerr << "error: Invalid Unicode argument\n"; return 1; }
        std::string value(static_cast<size_t>(size), '\0');
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, value.data(), size, nullptr, nullptr);
        value.pop_back();
        args.push_back(std::move(value));
    }
    return run(args);
}
#else
int main(int argc, char ** argv) { return run(std::vector<std::string>(argv, argv + argc)); }
#endif
