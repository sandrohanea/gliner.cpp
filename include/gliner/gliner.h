#ifndef GLINER_H
#define GLINER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// The context owns GGUF metadata and immutable model weights. A state owns inference
// scratch space and results. Keep the context alive until all its states are freed.
struct gliner_context;
struct gliner_state;

enum gliner_status {
    GLINER_STATUS_OK = 0,
    GLINER_STATUS_INVALID_ARGUMENT = 1,
    GLINER_STATUS_MODEL_ERROR = 2,
    GLINER_STATUS_BACKEND_ERROR = 3,
};

// Sequential loader, matching whisper.cpp's read/eof/close pattern.
// All callbacks are required. read may return fewer bytes than requested.
// Returning 0 ends the read: eof distinguishes truncation from a read failure.
// Callbacks must not throw or re-enter initialization. No seek is required.
typedef struct gliner_model_loader {
    void * context;
    size_t (*read)(void * context, void * output, size_t read_size);
    bool (*eof)(void * context);
    void (*close)(void * context);
} gliner_model_loader;

struct gliner_score {
    float logit;
    float probability; // sigmoid(logit), without task-specific calibration
};

struct gliner_text_params {
    int n_threads; // CPU threads; ignored on GPUs, but must be positive
    int max_tokens; // hard limit including schema; exceeding it is an error, never silent truncation
    int max_words;  // text word cap matching Python max_len; 0 means no word truncation
    const char * prompt; // optional task prompt
    const char * const * label_descriptions; // optional n_labels entries, in label order; NULL entries omitted
};

struct gliner_text_params gliner_default_text_params(void);

struct gliner_classification_task {
    const char * task; // nonempty task name, unique within a batch
    const char * const * labels; // n_labels nonempty names, unique within this task
    int n_labels;
    const char * prompt; // optional task prompt
    const char * const * label_descriptions; // optional n_labels entries; NULL entries omitted
};

struct gliner_batch_params {
    int n_threads; // CPU threads; ignored on GPUs, but must be positive
    int max_tokens; // hard limit across ALL schemas plus the shared text (1..4096)
    int max_words; // shared text word cap; 0 means no word truncation
};

struct gliner_batch_params gliner_default_batch_params(void);

struct gliner_task_result {
    int score_offset; // start in gliner_get_scores, in task order then label order
    int n_scores;
};

struct gliner_span_label {
    const char * name; // unique, nonempty extraction label/group
    const char * description; // optional description; empty string is distinct from NULL
};

struct gliner_span_params {
    int n_threads;
    int max_tokens; // combined schema + text limit, 1..4096; no automatic chunking
    int max_words; // 0 means no word truncation
    float threshold; // sigmoid probability cutoff, inclusive, 0..1
    int max_spans_per_label; // 0 means unlimited; applied after overlap suppression
    int allow_overlap; // 0: confidence-first greedy nonoverlap per label; 1: allow
};

struct gliner_span {
    int label_index;
    size_t start; // UTF-8 byte offsets into caller text, half-open [start, end)
    size_t end;
    float logit;
    float probability;
    const char * text; // exact source substring, owned by state
};

// Raw candidate diagnostics from the last successful extraction. End words are exclusive.
// logits is label-major [n_labels, n_candidates]; candidate coordinates index text words.
// Synthetic sentence punctuation is encoded but never returned as an extracted span.
struct gliner_span_scores {
    int n_labels;
    int n_candidates;
    int n_words;
    int max_width;
    int predicted_count; // upstream count gate: zero suppresses all entity results
    const int32_t * label_positions; // n_labels contextual [E] marker positions
    const int32_t * word_positions; // n_words first-subword positions in input_ids
    const int32_t * start_words; // n_candidates
    const int32_t * end_words; // n_candidates
    const float * logits;
    const float * count_logits; // 20 count classes
};

struct gliner_span_params gliner_default_span_params(void);

enum gliner_field_type {
    GLINER_FIELD_LIST = 0,
    GLINER_FIELD_SINGLE = 1,
};

struct gliner_record_field {
    const char * name;
    const char * description; // optional, encoded in field declaration order
    enum gliner_field_type type;
};

struct gliner_record_params {
    int n_threads;
    int max_tokens;
    int max_words;
    float threshold;
    int max_records; // safety limit, 1..19; exceeding it is an error, not truncation
    int max_spans_per_field; // list fields: 0 means all, after overlap suppression
    int allow_overlap; // 0 or 1, applied within each field and record
};

struct gliner_record {
    int slot_index; // original predicted slot; all-empty records are omitted
    int span_offset; // range in gliner_get_record_spans()
    int n_spans;
};

// Raw logits are [predicted_count, n_fields, n_candidates], in that order.
// A zero predicted count has no record logits; count_logits is always available.
struct gliner_record_scores {
    int n_fields;
    int n_candidates;
    int n_words;
    int max_width;
    int predicted_count;
    const int32_t * field_positions;
    const int32_t * word_positions;
    const int32_t * start_words;
    const int32_t * end_words;
    const float * logits;
    const float * count_logits; // 20 classes, counts 0..19
};

struct gliner_record_params gliner_default_record_params(void);

// Optional diagnostic callback: layer 0 is embeddings; 1..n_layers are encoder outputs.
// Values are row-major [n_tokens, hidden_size], valid only during the callback.
// Called synchronously on the scoring thread. Do not re-enter this state from the callback.
typedef void (*gliner_eval_callback)(int layer, const float * values, int n_tokens, int hidden_size, void * user_data);
void gliner_set_eval_callback(struct gliner_state * state, gliner_eval_callback callback, void * user_data);

// All initializers synchronously load an owned model and return NULL on failure.
// gliner_last_error() describes the failure until the next init/inference call.
// The caller still creates execution states with gliner_init_state().
struct gliner_context * gliner_init_from_file(const char * gguf_path);
// buffer is borrowed only during this call; it may be freed/unpinned afterwards.
struct gliner_context * gliner_init_from_buffer(const void * buffer, size_t buffer_size);
// Reads from the stream's current position. Once validated, close is called exactly
// once before returning, on success or failure. Invalid loaders invoke no callbacks.
// Neither the loader nor its context is retained. GGUF tensor data is streamed in
// bounded chunks, including unused tensors; no temporary model file is created.
struct gliner_context * gliner_init(struct gliner_model_loader * loader);
struct gliner_state * gliner_init_state(struct gliner_context * ctx);
void gliner_free(struct gliner_context * ctx);
void gliner_free_state(struct gliner_state * state);
const char * gliner_last_error(void);

// Model metadata. A NULL context returns 0.
int gliner_model_hidden_size(const struct gliner_context * ctx);
int gliner_model_n_tensors(const struct gliner_context * ctx);
int gliner_model_n_layers(const struct gliner_context * ctx);
int gliner_model_supports_text(const struct gliner_context * ctx);
int gliner_model_supports_spans(const struct gliner_context * ctx);
// Fixed by GGML_CUDA/GGML_METAL at build time, not by an initialization parameter.
// Returns "cpu", "cuda" or "metal" without initializing any devices.
const char * gliner_build_backend(void);
// Read-only runtime identity. NULL context returns NULL; strings live until context free.
const char * gliner_model_backend_name(const struct gliner_context * ctx); // "cpu", "cuda", "metal"
const char * gliner_model_device_name(const struct gliner_context * ctx);

// Score precomputed contextual [L] marker states. `label_states` is row-major
// [n_labels, hidden_size].
// n_threads controls CPU execution only and is ignored by GPU backends (must still be positive).
// Each state may be used by only one thread at a time; distinct states can run
// concurrently against the same context. Returns a gliner_status value.
int gliner_score_label_states(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const float * label_states,
    int n_labels,
    int n_threads);

// UTF-8 text classification for one task. Labels must be nonempty and unique.
// Literal [SEP_TEXT]/[SEP_STRUCT] task and label names are rejected.
// params may be NULL for defaults. Scores retain independent sigmoid probabilities;
// exclusive softmax or multi-label decisions can be derived from the raw logits.
int gliner_classify_text(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const char * text,
    const char * task,
    const char * const * labels,
    int n_labels,
    const struct gliner_text_params * params);

// Joint classification: one text, ordered tasks, one encoder pass. Schemas are
// separated by [SEP_STRUCT] and the text is included once, matching upstream.
// Tasks attend to one another, so scores can differ from separate single-task calls.
// Input pointers are borrowed only during the call. params may be NULL for defaults.
// A one-task batch is equivalent to gliner_classify_text. On failure all results clear.
int gliner_classify_text_batch(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const char * text,
    const struct gliner_classification_task * tasks,
    int n_tasks,
    const struct gliner_batch_params * params);

// Task ranges for the last successful classification call. Single-task calls expose one range.
// NULL state, states-only calls and failed calls return 0/NULL. Results have the
// same lifetime as scores. Repeated label names across different tasks are allowed.
int gliner_n_tasks(const struct gliner_state * state);
const struct gliner_task_result * gliner_get_task_results(const struct gliner_state * state);

// Grouped, verbatim entity/keyphrase extraction in one encoder pass, using upstream
// entity schema semantics. Scores do not guarantee retrieval usefulness. Pointers are
// borrowed only for this call; all inference results clear on failure or the next call.
// Requires a checkpoint converted with span metadata. params may be NULL for defaults.
int gliner_extract_spans(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const char * text,
    const struct gliner_span_label * labels,
    int n_labels,
    const struct gliner_span_params * params);

// Results ordered by label index, then descending confidence and source position.
// Classification/states-only calls and failed extraction clear these results.
int gliner_n_spans(const struct gliner_state * state);
const struct gliner_span * gliner_get_spans(const struct gliner_state * state);
const struct gliner_span_scores * gliner_get_span_scores(const struct gliner_state * state);

// Extract repeated records of one structure, without caller-supplied topic groups.
// All fields must have unique nonempty names; list fields select multiple spans,
// single fields select the best surviving span. Missing fields have no span entries.
// One encoder pass predicts the count; a second head-only graph conditions each slot.
// Requires the same markerV0/count_lstm metadata as gliner_extract_spans.
int gliner_extract_records(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const char * text,
    const char * structure,
    const struct gliner_record_field * fields,
    int n_fields,
    const struct gliner_record_params * params);

// State-owned results, valid until the next inference/free. All other inference
// operations clear these. label_index in record spans is the input FIELD index.
// Fields are decoded independently; records are not deduplicated across slots.
int gliner_n_records(const struct gliner_state * state);
const struct gliner_record * gliner_get_records(const struct gliner_state * state);
const struct gliner_span * gliner_get_record_spans(const struct gliner_state * state);
const struct gliner_record_scores * gliner_get_record_scores(const struct gliner_state * state);

// Token IDs from the last successful classification/extraction (empty after states-only or failure).
int gliner_n_tokens(const struct gliner_state * state);
const int32_t * gliner_get_token_ids(const struct gliner_state * state);
// Classification-only label diagnostics; extraction uses gliner_get_span_scores instead.
const int32_t * gliner_get_label_positions(const struct gliner_state * state); // n_scores entries
const float * gliner_get_label_states(const struct gliner_state * state); // n_scores * hidden_size

// All labels flattened in task order, then label order. Probabilities are independent
// sigmoid values, not a softmax across the batch. Normalize within each task if needed.
// Results belong to `state` and remain valid until its next inference call or free.
// A failed score call clears previous results. NULL state returns 0/NULL.
int gliner_n_scores(const struct gliner_state * state);
const struct gliner_score * gliner_get_scores(const struct gliner_state * state);

#ifdef __cplusplus
}
#endif

#endif // GLINER_H
