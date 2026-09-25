#include "common.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>

namespace {
using namespace gliner_examples;
using Clock = std::chrono::steady_clock;
using Model = std::unique_ptr<gliner_context, decltype(&gliner_free)>;
using State = std::unique_ptr<gliner_state, decltype(&gliner_free_state)>;

struct Options {
    std::string model, text, text_file, mode = "both";
    std::vector<TaskOptions> tasks{1};
    gliner_batch_params params = gliner_default_batch_params();
    int warmup = 3;
    int iterations = 20;
};

struct Measurement {
    std::string mode;
    double state_init_ms = 0;
    double first_request_ms = 0;
    std::vector<double> samples_ms;
    std::vector<int> tokens_per_pass;
    std::vector<float> last_logits;
    size_t state_count = 0;
};

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void usage() {
    std::cout <<
        "Usage: gliner-bench --model model.gguf --text \"text\" --task intent --labels a,b\n"
        "       gliner-bench --model model.gguf --text-file input.txt --task intent --labels a,b --task sentiment --labels positive,negative\n"
        "       gliner-bench --build-info\n"
        "Options:\n"
        "  --mode joint|separate|both  Default both; compare one joint pass vs one pass per task\n"
        "  --warmup N                 Warmup requests after the first request (default 3, may be 0)\n"
        "  --iterations N             Measured requests per mode (default 20, must be positive)\n"
        "  --threads N                CPU threads (default 1; ignored on GPUs)\n"
        "  --max-words N              Text word cap (default 0 = unlimited)\n"
        "  --max-tokens N             Per-pass token limit, including schema (default 512, 1..4096)\n"
        "  --task NAME                Start a task group; names must be unique\n"
        "  --labels a,b | --label NAME Candidate labels; --label can be repeated\n"
        "  --prompt TEXT              Current task's prompt\n"
        "  --description LABEL=TEXT   Current task's label description; repeat as needed\n"
        "Outputs JSON with raw samples, median/p95 latency and throughput.\n"
        "One request answers ALL tasks. Model is loaded once. No decoding or JSON output is timed.\n"
        "Backend is fixed at build time. Joint and separate logits need not match.\n";
}

Options parse(const std::vector<std::string> & args) {
    Options options;
    bool has_text = false, has_file = false, has_task = false;
    for (size_t i = 1; i < args.size(); ++i) {
        const auto & arg = args[i];
        if (arg == "--backend" || arg == "--device" || arg == "--list-devices") {
            throw std::invalid_argument("Backend selection is build-time only; configure GGML_CUDA/GGML_METAL with CMake");
        }
        if (i + 1 == args.size()) throw std::invalid_argument("Missing value for " + arg);
        const auto & value = args[++i];
        auto & task = options.tasks.back();
        if (arg == "--model") options.model = value;
        else if (arg == "--text") { options.text = value; has_text = true; }
        else if (arg == "--text-file") { options.text_file = value; has_file = true; }
        else if (arg == "--mode") options.mode = value;
        else if (arg == "--warmup") options.warmup = integer(value);
        else if (arg == "--iterations") options.iterations = integer(value);
        else if (arg == "--threads") options.params.n_threads = integer(value);
        else if (arg == "--max-words") options.params.max_words = integer(value);
        else if (arg == "--max-tokens") options.params.max_tokens = integer(value);
        else if (arg == "--task") {
            if (has_task) options.tasks.emplace_back();
            options.tasks.back().name = value;
            has_task = true;
        } else if (arg == "--labels") { task.labels = split_labels(value); task.comma_labels = true; }
        else if (arg == "--label") { task.labels.push_back(value); task.repeated_labels = true; }
        else if (arg == "--prompt") task.prompt = value;
        else if (arg == "--description") task.description_args.push_back(value);
        else throw std::invalid_argument("Unknown benchmark option: " + arg);
    }
    if (options.model.empty()) throw std::invalid_argument("--model is required");
    if (has_text == has_file || (has_file && options.text_file.empty())) {
        throw std::invalid_argument("Provide exactly one of --text or --text-file");
    }
    if (options.mode != "joint" && options.mode != "separate" && options.mode != "both") {
        throw std::invalid_argument("--mode must be joint, separate or both");
    }
    if (options.iterations <= 0 || options.params.n_threads <= 0 ||
        options.params.max_tokens <= 0 || options.params.max_tokens > 4096) {
        throw std::invalid_argument("Iterations and threads must be positive; max_tokens must be 1..4096");
    }
    size_t count = 0;
    for (size_t t = 0; t < options.tasks.size(); ++t) {
        auto & task = options.tasks[t];
        task.validate(true);
        for (size_t previous = 0; previous < t; ++previous) {
            if (options.tasks[previous].name == task.name) throw std::invalid_argument("Task names must be unique");
        }
        if (task.labels.size() > static_cast<size_t>(options.params.max_tokens)) {
            throw std::invalid_argument("Task labels must fit max_tokens");
        }
        if (task.labels.size() > static_cast<size_t>(std::numeric_limits<int>::max()) - count) {
            throw std::invalid_argument("Too many labels");
        }
        count += task.labels.size();
    }
    if (options.mode != "separate" && count > static_cast<size_t>(options.params.max_tokens)) {
        throw std::invalid_argument("Joint task labels must fit max_tokens");
    }
    if (has_file) options.text = read_text_file(options.text_file);
    if (options.text.find('\0') != std::string::npos) throw std::invalid_argument("Text contains NUL");
    return options;
}

Measurement measure(gliner_context * model, const Options & options,
                    const std::vector<gliner_classification_task> & tasks, bool joint) {
    Measurement result;
    result.mode = joint ? "joint" : "separate";
    result.state_count = joint ? 1 : tasks.size();
    result.samples_ms.reserve(static_cast<size_t>(options.iterations));
    std::vector<State> states;
    states.reserve(result.state_count);
    auto start = Clock::now();
    for (size_t i = 0; i < result.state_count; ++i) {
        states.emplace_back(gliner_init_state(model), gliner_free_state);
        if (!states.back()) throw std::runtime_error(gliner_last_error());
    }
    result.state_init_ms = elapsed_ms(start);

    auto request = [&] {
        // Public inference calls synchronize the backend and retrieve scores before returning.
        if (joint) {
            if (gliner_classify_text_batch(model, states[0].get(), options.text.c_str(), tasks.data(),
                                          static_cast<int>(tasks.size()), &options.params) != GLINER_STATUS_OK) {
                throw std::runtime_error(gliner_last_error());
            }
        } else {
            for (size_t i = 0; i < tasks.size(); ++i) {
                const auto & task = tasks[i];
                gliner_text_params params = {options.params.n_threads, options.params.max_tokens,
                                             options.params.max_words, task.prompt, task.label_descriptions};
                if (gliner_classify_text(model, states[i].get(), options.text.c_str(), task.task,
                                        task.labels, task.n_labels, &params) != GLINER_STATUS_OK) {
                    throw std::runtime_error(gliner_last_error());
                }
            }
        }
    };
    auto collect = [&](bool first) {
        result.last_logits.clear();
        for (size_t s = 0; s < states.size(); ++s) {
            auto * state = states[s].get();
            const int tokens = gliner_n_tokens(state);
            if (tokens <= 0) throw std::runtime_error("Inference returned no tokens");
            if (first) result.tokens_per_pass.push_back(tokens);
            else if (tokens != result.tokens_per_pass[s]) throw std::runtime_error("Token count changed between repetitions");
            const int expected_tasks = joint ? static_cast<int>(tasks.size()) : 1;
            if (gliner_n_tasks(state) != expected_tasks) throw std::runtime_error("Inference returned an unexpected task count");
            const auto * ranges = gliner_get_task_results(state);
            int expected_scores = 0;
            for (int t = 0; t < expected_tasks; ++t) {
                const auto & task = tasks[joint ? static_cast<size_t>(t) : s];
                if (!ranges || ranges[t].score_offset != expected_scores || ranges[t].n_scores != task.n_labels) {
                    throw std::runtime_error("Inference returned unexpected task score ranges");
                }
                expected_scores += task.n_labels;
            }
            const auto * scores = gliner_get_scores(state);
            if (!scores || gliner_n_scores(state) != expected_scores) throw std::runtime_error("Inference returned unexpected scores");
            for (int i = 0; i < expected_scores; ++i) {
                if (!std::isfinite(scores[i].logit)) throw std::runtime_error("Inference returned nonfinite logits");
                result.last_logits.push_back(scores[i].logit);
            }
        }
    };
    start = Clock::now();
    request();
    result.first_request_ms = elapsed_ms(start);
    collect(true);
    for (int i = 0; i < options.warmup; ++i) {
        request();
        collect(false);
    }
    for (int i = 0; i < options.iterations; ++i) {
        start = Clock::now();
        request();
        const double ms = elapsed_ms(start);
        if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("Nonpositive or nonfinite benchmark duration");
        result.samples_ms.push_back(ms);
        collect(false);
    }
    return result;
}

double percentile(const std::vector<double> & sorted, double p) {
    const double rank = (sorted.size() - 1) * p;
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

void write_measurement(std::ostream & out, const Measurement & result, size_t n_tasks) {
    auto sorted = result.samples_ms;
    std::sort(sorted.begin(), sorted.end());
    const double total = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    const double mean = total / sorted.size();
    const uint64_t tokens = std::accumulate(result.tokens_per_pass.begin(), result.tokens_per_pass.end(), uint64_t(0));
    out << "{\"mode\":\"" << result.mode << "\",\"state_count\":" << result.state_count
        << ",\"encoder_passes_per_request\":" << result.tokens_per_pass.size()
        << ",\"tokens_per_pass\":";
    array(out, result.tokens_per_pass.data(), result.tokens_per_pass.size());
    out << ",\"tokens_per_request\":" << tokens
        << ",\"state_init_ms\":" << result.state_init_ms
        << ",\"first_request_ms\":" << result.first_request_ms
        << ",\"measured_total_ms\":" << total
        << ",\"latency_ms\":{\"min\":" << sorted.front() << ",\"mean\":" << mean
        << ",\"median\":" << percentile(sorted, 0.5) << ",\"p95\":" << percentile(sorted, 0.95)
        << ",\"max\":" << sorted.back() << "}"
        << ",\"requests_per_second\":" << 1000.0 / mean
        << ",\"questions_per_second\":" << 1000.0 * n_tasks / mean
        << ",\"encoded_tokens_per_second\":" << 1000.0 * tokens / mean
        << ",\"samples_ms\":";
    array(out, result.samples_ms.data(), result.samples_ms.size());
    out << ",\"last_logits\":";
    array(out, result.last_logits.data(), result.last_logits.size());
    out << '}';
}

void environment_value(std::ostream & out, const char * name) {
    const char * value = std::getenv(name);
    if (value) out << '"' << escape_json(value) << '"';
    else out << "null";
}

} // namespace

int gliner_cli_main(const std::vector<std::string> & args) {
    try {
        if (args.size() == 2 && (args[1] == "--help" || args[1] == "-h")) { usage(); return 0; }
        if (args.size() == 2 && args[1] == "--build-info") {
            std::cout << "{\"backend\":\"" << gliner_build_backend() << "\"}\n";
            return 0;
        }
        auto options = parse(args);
        if (std::string(gliner_build_backend()) == "cuda") {
            const char * tf32 = std::getenv("NVIDIA_TF32_OVERRIDE");
            const char * compute = std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
            if (!tf32 || std::string(tf32) != "0" || !compute || std::string(compute) != "f32") {
                std::cerr << "warning: CUDA parity settings are not set: use NVIDIA_TF32_OVERRIDE=0 and "
                             "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 before launch for comparable precision\n";
            }
        }
#ifndef NDEBUG
        std::cerr << "warning: use a Release build for representative benchmark results\n";
#endif
        std::vector<gliner_classification_task> tasks;
        for (auto & task : options.tasks) tasks.push_back(task.input());
        const auto file_bytes = std::filesystem::file_size(std::filesystem::u8path(options.model));
        const auto start = Clock::now();
        Model model(gliner_init_from_file(options.model.c_str()), gliner_free);
        const double load_ms = elapsed_ms(start);
        if (!model) throw std::runtime_error(gliner_last_error());
        if (!gliner_model_supports_text(model.get())) throw std::runtime_error("Benchmark requires a text-capable GGUF; reconvert the checkpoint");

        std::vector<Measurement> results;
        if (options.mode != "separate") results.push_back(measure(model.get(), options, tasks, true));
        if (options.mode != "joint") results.push_back(measure(model.get(), options, tasks, false));

        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(17)
            << "{\"schema_version\":1,\"backend\":\"" << gliner_model_backend_name(model.get())
            << "\",\"device\":\"" << escape_json(gliner_model_device_name(model.get()))
            << "\",\"build_type\":\"" << escape_json(GLINER_BENCH_BUILD_TYPE)
            << "\",\"compiler\":\"" << escape_json(GLINER_BENCH_COMPILER)
            << "\",\"model\":\"" << escape_json(options.model) << "\",\"model_file_bytes\":" << file_bytes
            << ",\"hidden_size\":" << gliner_model_hidden_size(model.get())
            << ",\"encoder_layers\":" << gliner_model_n_layers(model.get())
            << ",\"model_load_count\":1,\"model_load_ms\":" << load_ms
            << ",\"threads\":" << options.params.n_threads
            << ",\"max_tokens\":" << options.params.max_tokens << ",\"max_words\":" << options.params.max_words
            << ",\"text_bytes\":" << options.text.size() << ",\"task_count\":" << tasks.size()
            << ",\"warmup_requests\":" << options.warmup << ",\"iterations\":" << options.iterations
            << ",\"percentile_method\":\"linear interpolation at (n-1)*p\""
            << ",\"timing_scope\":\"synchronous text classification APIs; one request answers all tasks\""
            << ",\"precision_environment\":{\"NVIDIA_TF32_OVERRIDE\":";
        environment_value(out, "NVIDIA_TF32_OVERRIDE");
        out << ",\"GGML_CUDA_CUBLAS_COMPUTE_TYPE\":";
        environment_value(out, "GGML_CUDA_CUBLAS_COMPUTE_TYPE");
        out << "},\"tasks\":[";
        for (size_t i = 0; i < tasks.size(); ++i) {
            if (i) out << ',';
            const auto & task = options.tasks[i];
            out << "{\"task\":\"" << escape_json(task.name) << "\",\"labels\":[";
            for (size_t j = 0; j < task.labels.size(); ++j) {
                if (j) out << ',';
                out << '"' << escape_json(task.labels[j]) << '"';
            }
            out << "],\"prompt\":\"" << escape_json(task.prompt) << "\",\"label_descriptions\":[";
            for (size_t j = 0; j < task.descriptions.size(); ++j) {
                if (j) out << ',';
                if (task.descriptions[j]) out << '"' << escape_json(*task.descriptions[j]) << '"';
                else out << "null";
            }
            out << "]}";
        }
        out << "],\"results\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i) out << ',';
            write_measurement(out, results[i], tasks.size());
        }
        out << "]}\n";
        std::cout << out.str();
        if (!std::cout) throw std::runtime_error("Cannot write benchmark output");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
