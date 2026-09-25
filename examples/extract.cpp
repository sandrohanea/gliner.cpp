#include "common.h"

#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>

namespace {
using namespace gliner_examples;

void usage() {
    std::cout <<
        "Usage: gliner-extract --model model.gguf --text \"text\" --labels product,person\n"
        "       gliner-extract --model model.gguf --text-file input.txt --label \"D1 search term\" --description \"D1 search term=Terms about D1 limits\"\n"
        "Options:\n"
        "  --labels a,b | --label NAME  Extraction labels/groups; --label may repeat\n"
        "  --description NAME=TEXT      Optional label description; may repeat\n"
        "  --threshold P               Inclusive probability cutoff (default 0.5)\n"
        "  --top-k N                   Maximum spans per label after suppression (0 = all)\n"
        "  --allow-overlap             Keep overlapping spans within a label\n"
        "  --threads N                 CPU threads (default 1)\n"
        "  --max-tokens N              Schema + text limit, 1..4096 (default 512)\n"
        "  --max-words N               Explicit word truncation (0 = none)\n"
        "  --debug                     Include tokens, word/label positions, count and raw span logits\n"
        "Returns verbatim UTF-8 spans and byte offsets. No generated terms or automatic long-text chunking.\n";
}
}

int gliner_cli_main(const std::vector<std::string> & args) {
    try {
        if (args.size() == 2 && (args[1] == "--help" || args[1] == "-h")) { usage(); return 0; }
        if (args.size() == 2 && args[1] == "--build-info") {
            std::cout << "{\"backend\":\"" << gliner_build_backend() << "\"}\n";
            return 0;
        }
        std::string model_path, text, text_file;
        bool has_text = false, has_file = false, debug = false;
        auto params = gliner_default_span_params();
        TaskOptions schema;
        schema.name = "entities";
        for (size_t i = 1; i < args.size(); ++i) {
            const auto & arg = args[i];
            if (arg == "--debug") { debug = true; continue; }
            if (arg == "--allow-overlap") { params.allow_overlap = 1; continue; }
            if (arg == "--backend" || arg == "--device") throw std::invalid_argument("Backend selection is build-time only");
            if (i + 1 == args.size()) throw std::invalid_argument("Missing value for " + arg);
            const auto & value = args[++i];
            if (arg == "--model") model_path = value;
            else if (arg == "--text") { text = value; has_text = true; }
            else if (arg == "--text-file") { text_file = value; has_file = true; }
            else if (arg == "--labels") { schema.labels = split_labels(value); schema.comma_labels = true; }
            else if (arg == "--label") { schema.labels.push_back(value); schema.repeated_labels = true; }
            else if (arg == "--description") schema.description_args.push_back(value);
            else if (arg == "--threads") params.n_threads = integer(value);
            else if (arg == "--max-tokens") params.max_tokens = integer(value);
            else if (arg == "--max-words") params.max_words = integer(value);
            else if (arg == "--threshold") {
                const double threshold = number(value);
                if (threshold < 0 || threshold > 1) throw std::invalid_argument("Threshold must be in [0,1]");
                params.threshold = static_cast<float>(threshold);
            } else if (arg == "--top-k") params.max_spans_per_label = integer(value);
            else throw std::invalid_argument("Unknown extraction option: " + arg);
        }
        if (model_path.empty() || has_text == has_file) throw std::invalid_argument("Provide --model and exactly one of --text or --text-file");
        if (params.n_threads <= 0 || params.max_tokens <= 0 || params.max_tokens > 4096) {
            throw std::invalid_argument("Threads must be positive and max_tokens must be 1..4096");
        }
        schema.validate(true);
        if (schema.labels.size() > static_cast<size_t>(params.max_tokens)) throw std::invalid_argument("Labels must fit max_tokens");
        if (has_file) text = read_text_file(text_file);
        if (text.find('\0') != std::string::npos) throw std::invalid_argument("Text contains NUL");
        std::vector<gliner_span_label> labels;
        for (size_t i = 0; i < schema.labels.size(); ++i) {
            labels.push_back({schema.labels[i].c_str(), schema.descriptions[i] ? schema.descriptions[i]->c_str() : nullptr});
        }
        std::unique_ptr<gliner_context, decltype(&gliner_free)> model(gliner_init_from_file(model_path.c_str()), gliner_free);
        if (!model) throw std::runtime_error(gliner_last_error());
        std::unique_ptr<gliner_state, decltype(&gliner_free_state)> state(gliner_init_state(model.get()), gliner_free_state);
        if (!state) throw std::runtime_error(gliner_last_error());
        if (gliner_extract_spans(model.get(), state.get(), text.c_str(), labels.data(), static_cast<int>(labels.size()), &params) != GLINER_STATUS_OK) {
            throw std::runtime_error(gliner_last_error());
        }
        const auto * spans = gliner_get_spans(state.get());
        const auto * raw = gliner_get_span_scores(state.get());
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << std::setprecision(9) << "{\"groups\":[";
        for (size_t label = 0; label < labels.size(); ++label) {
            if (label) output << ',';
            output << "{\"label\":\"" << escape_json(schema.labels[label]) << "\",\"spans\":[";
            bool first = true;
            for (int i = 0; i < gliner_n_spans(state.get()); ++i) {
                const auto & span = spans[i];
                if (span.label_index != static_cast<int>(label)) continue;
                if (!first) output << ',';
                output << "{\"text\":\"" << escape_json(span.text) << "\",\"start\":" << span.start << ",\"end\":" << span.end
                       << ",\"logit\":" << span.logit << ",\"probability\":" << span.probability << '}';
                first = false;
            }
            output << "]}";
        }
        output << "],\"offset_unit\":\"utf8_bytes\",\"max_span_width\":" << raw->max_width;
        if (debug) {
            output << ",\"backend\":\"" << gliner_model_backend_name(model.get()) << "\",\"input_ids\":";
            array(output, gliner_get_token_ids(state.get()), static_cast<size_t>(gliner_n_tokens(state.get())));
            output << ",\"label_positions\":";
            array(output, raw->label_positions, raw->n_labels);
            output << ",\"word_positions\":";
            array(output, raw->word_positions, raw->n_words);
            output << ",\"start_words\":";
            array(output, raw->start_words, raw->n_candidates);
            output << ",\"end_words\":";
            array(output, raw->end_words, raw->n_candidates);
            output << ",\"span_logits\":";
            array(output, raw->logits, static_cast<size_t>(raw->n_candidates) * raw->n_labels);
            output << ",\"count_logits\":";
            array(output, raw->count_logits, 20);
            output << ",\"predicted_count\":" << raw->predicted_count;
        }
        output << "}\n";
        std::cout << output.str();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
