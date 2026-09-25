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
        "       gliner-extract --model model.gguf --text-file input.txt --structure search_requests --field terms\n"
        "Options:\n"
        "  --labels a,b | --label NAME  Extraction labels/groups; --label may repeat\n"
        "  --description NAME=TEXT      Optional label description; may repeat\n"
        "  --structure NAME            Extract repeated records instead of entity groups\n"
        "  --field NAME                List-valued field; repeat for additional fields\n"
        "  --single-field NAME         Field containing only its best surviving span\n"
        "  --record-mode MODE          Boundary records: anchorless (default) or latent\n"
        "  --max-records N             Safety limit (default 19); span <=19, boundary <=4096\n"
        "  --threshold P               Inclusive probability cutoff (default 0.5)\n"
        "  --top-k N                   Maximum spans per label/list field after suppression (0 = all)\n"
        "  --allow-overlap             Keep overlapping spans within a label\n"
        "  --threads N                 CPU threads (default 1)\n"
        "  --max-tokens N              Schema + text limit, 1..4096 (default 512)\n"
        "  --max-words N               Explicit word truncation (0 = none)\n"
        "  --debug                     Include tokens, routing and architecture-specific raw scores\n"
        "Returns verbatim UTF-8 spans and byte offsets. No generated terms or automatic long-text chunking.\n";
}

void write_span(std::ostream & out, const gliner_span & span) {
    out << "{\"text\":\"" << escape_json(span.text) << "\",\"start\":" << span.start << ",\"end\":" << span.end
        << ",\"logit\":" << span.logit << ",\"probability\":" << span.probability << '}';
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
        bool has_labels = false, has_structure = false, has_record_limit = false;
        gliner_record_mode record_mode = GLINER_RECORD_AUTO;
        int max_records = 19;
        std::vector<gliner_field_type> field_types;
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
            else if (arg == "--labels") { schema.labels = split_labels(value); schema.comma_labels = true; has_labels = true; }
            else if (arg == "--label") { schema.labels.push_back(value); schema.repeated_labels = true; has_labels = true; }
            else if (arg == "--structure") {
                if (has_structure || value.empty() || value == "entities") throw std::invalid_argument("Provide one nonempty --structure name, other than 'entities'");
                schema.name = value;
                has_structure = true;
            } else if (arg == "--field" || arg == "--single-field") {
                schema.labels.push_back(value);
                field_types.push_back(arg == "--field" ? GLINER_FIELD_LIST : GLINER_FIELD_SINGLE);
            } else if (arg == "--max-records") { max_records = integer(value); has_record_limit = true; }
            else if (arg == "--record-mode") {
                if (value == "anchorless") record_mode = GLINER_RECORD_ANCHORLESS;
                else if (value == "latent") record_mode = GLINER_RECORD_LATENT;
                else throw std::invalid_argument("Record mode must be anchorless or latent");
            }
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
        if ((has_structure && (has_labels || field_types.empty())) ||
            (!has_structure && (!field_types.empty() || has_record_limit || record_mode != GLINER_RECORD_AUTO)) || max_records < 1 || max_records > 4096) {
            throw std::invalid_argument("Use --structure with fields and max-records 1..4096 (span <=19), or entity labels; do not mix the modes");
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
        int status;
        if (has_structure) {
            std::vector<gliner_record_field> fields;
            for (size_t i = 0; i < labels.size(); ++i) fields.push_back({labels[i].name, labels[i].description, field_types[i]});
            const gliner_record_params record_params = {params.n_threads, params.max_tokens, params.max_words,
                params.threshold, max_records, params.max_spans_per_label, params.allow_overlap, record_mode};
            status = gliner_extract_records(model.get(), state.get(), text.c_str(), schema.name.c_str(),
                                           fields.data(), static_cast<int>(fields.size()), &record_params);
        } else {
            status = gliner_extract_spans(model.get(), state.get(), text.c_str(), labels.data(), static_cast<int>(labels.size()), &params);
        }
        if (status != GLINER_STATUS_OK) {
            throw std::runtime_error(gliner_last_error());
        }
        if (has_structure) {
            const auto * records = gliner_get_records(state.get());
            const auto * spans = gliner_get_record_spans(state.get());
            const auto * raw = gliner_get_record_scores(state.get());
            const auto * boundary = gliner_get_boundary_record_scores(state.get());
            std::ostringstream output;
            output.imbue(std::locale::classic());
            output << std::setprecision(9) << "{\"structure\":\"" << escape_json(schema.name) << '"';
            if (boundary) {
                output << ",\"record_count\":" << gliner_n_records(state.get()) << ",\"record_mode\":\""
                       << (boundary->mode == GLINER_RECORD_LATENT ? "latent" : "anchorless") << '"';
            } else output << ",\"predicted_count\":" << raw->predicted_count;
            output << ",\"records\":[";
            for (int r = 0; r < gliner_n_records(state.get()); ++r) {
                if (r) output << ',';
                output << "{\"slot_index\":" << records[r].slot_index << ",\"fields\":{";
                for (size_t f = 0; f < labels.size(); ++f) {
                    if (f) output << ',';
                    output << '"' << escape_json(schema.labels[f]) << "\":";
                    const bool list = field_types[f] == GLINER_FIELD_LIST;
                    if (list) output << '[';
                    bool first = true;
                    for (int i = records[r].span_offset; i < records[r].span_offset + records[r].n_spans; ++i) {
                        if (spans[i].label_index != static_cast<int>(f)) continue;
                        if (!first) output << ',';
                        write_span(output, spans[i]);
                        first = false;
                    }
                    if (list) output << ']';
                    else if (first) output << "null";
                }
                output << "}}";
            }
            output << "],\"offset_unit\":\"utf8_bytes\",\"max_span_width\":";
            if (boundary) output << "null"; else output << raw->max_width;
            if (debug) {
                output << ",\"backend\":\"" << gliner_model_backend_name(model.get()) << "\",\"input_ids\":";
                array(output, gliner_get_token_ids(state.get()), static_cast<size_t>(gliner_n_tokens(state.get())));
                output << ",\"field_positions\":";
                array(output, boundary ? boundary->field_positions : raw->field_positions, labels.size());
                output << ",\"word_positions\":";
                array(output, boundary ? boundary->word_positions : raw->word_positions, boundary ? boundary->n_words : raw->n_words);
                output << ",\"start_words\":";
                array(output, boundary ? boundary->start_words : raw->start_words, boundary ? boundary->n_candidates : raw->n_candidates);
                output << ",\"end_words\":";
                array(output, boundary ? boundary->end_words : raw->end_words, boundary ? boundary->n_candidates : raw->n_candidates);
                if (boundary) {
                    output << ",\"architecture\":\"boundary\",\"n_instances\":" << boundary->n_instances
                           << ",\"record_temperature\":" << boundary->temperature << ",\"pair_logits\":";
                    array(output, boundary->pair_logits, static_cast<size_t>(boundary->n_fields) * boundary->n_candidates);
                    output << ",\"object_logits\":";
                    array(output, boundary->object_logits, boundary->n_instances);
                    output << ",\"assignment_logits\":";
                    array(output, boundary->assignment_logits,
                          static_cast<size_t>(boundary->n_fields) * boundary->n_instances * (boundary->n_candidates + 1));
                } else {
                    output << ",\"record_logits\":";
                    array(output, raw->logits, static_cast<size_t>(raw->predicted_count) * raw->n_fields * raw->n_candidates);
                    output << ",\"count_logits\":";
                    array(output, raw->count_logits, 20);
                }
            }
            output << "}\n";
            std::cout << output.str();
            return 0;
        }
        const auto * spans = gliner_get_spans(state.get());
        const auto * raw = gliner_get_span_scores(state.get());
        const auto * boundary = gliner_get_boundary_scores(state.get());
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
                write_span(output, span);
                first = false;
            }
            output << "]}";
        }
        output << "],\"offset_unit\":\"utf8_bytes\",\"max_span_width\":";
        if (boundary) output << "null";
        else output << raw->max_width;
        if (debug) {
            output << ",\"backend\":\"" << gliner_model_backend_name(model.get()) << "\",\"input_ids\":";
            array(output, gliner_get_token_ids(state.get()), static_cast<size_t>(gliner_n_tokens(state.get())));
            output << ",\"label_positions\":";
            array(output, boundary ? boundary->label_positions : raw->label_positions, labels.size());
            output << ",\"word_positions\":";
            array(output, boundary ? boundary->word_positions : raw->word_positions, boundary ? boundary->n_words : raw->n_words);
            output << ",\"start_words\":";
            array(output, boundary ? boundary->start_words : raw->start_words, boundary ? boundary->n_candidates : raw->n_candidates);
            output << ",\"end_words\":";
            array(output, boundary ? boundary->end_words : raw->end_words, boundary ? boundary->n_candidates : raw->n_candidates);
            output << ",\"span_logits\":";
            array(output, boundary ? boundary->logits : raw->logits,
                  labels.size() * static_cast<size_t>(boundary ? boundary->n_candidates : raw->n_candidates));
            if (boundary) {
                output << ",\"architecture\":\"boundary\",\"pool_capacity\":" << boundary->pool_capacity
                       << ",\"pair_temperature\":" << boundary->pair_temperature
                       << ",\"abstention_threshold\":" << boundary->abstention_threshold << ",\"null_logits\":";
                array(output, boundary->null_logits, boundary->n_labels);
                output << ",\"start_logits\":";
                array(output, boundary->start_logits, static_cast<size_t>(boundary->n_labels) * (boundary->n_words + 1));
                output << ",\"end_logits\":";
                array(output, boundary->end_logits, static_cast<size_t>(boundary->n_labels) * (boundary->n_words + 1));
                output << ",\"proposal_logits\":";
                array(output, boundary->proposal_logits, boundary->n_candidates);
            } else {
                output << ",\"count_logits\":";
                array(output, raw->count_logits, 20);
                output << ",\"predicted_count\":" << raw->predicted_count;
            }
        }
        output << "}\n";
        std::cout << output.str();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
