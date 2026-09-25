#ifndef GLINER_H
#define GLINER_H

#include <stddef.h>
#include <stdint.h>

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

struct gliner_score {
    float logit;
    float probability; // sigmoid(logit), without task-specific calibration
};

struct gliner_text_params {
    int n_threads;
    int max_tokens; // hard limit including schema; exceeding it is an error, never silent truncation
    int max_words;  // text word cap matching Python max_len; 0 means no word truncation
    const char * prompt; // optional task prompt
    const char * const * label_descriptions; // optional n_labels entries, in label order; NULL entries omitted
};

struct gliner_text_params gliner_default_text_params(void);

// Optional diagnostic callback: layer 0 is embeddings; 1..n_layers are encoder outputs.
// Values are row-major [n_tokens, hidden_size], valid only during the callback.
// Called synchronously on the scoring thread. Do not re-enter this state from the callback.
typedef void (*gliner_eval_callback)(int layer, const float * values, int n_tokens, int hidden_size, void * user_data);
void gliner_set_eval_callback(struct gliner_state * state, gliner_eval_callback callback, void * user_data);

// Returns NULL on failure. gliner_last_error() explains the last failure on
// the calling thread; its pointer is valid until the next init or score call there.
struct gliner_context * gliner_init_from_file(const char * gguf_path);
struct gliner_state * gliner_init_state(struct gliner_context * ctx);
void gliner_free(struct gliner_context * ctx);
void gliner_free_state(struct gliner_state * state);
const char * gliner_last_error(void);

// Model metadata. A NULL context returns 0.
int gliner_model_hidden_size(const struct gliner_context * ctx);
int gliner_model_n_tensors(const struct gliner_context * ctx);
int gliner_model_n_layers(const struct gliner_context * ctx);
int gliner_model_supports_text(const struct gliner_context * ctx);

// Score precomputed contextual [L] marker states. `label_states` is row-major
// [n_labels, hidden_size].
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

// Diagnostics from the last successful text call (empty after a states-only call or failure).
int gliner_n_tokens(const struct gliner_state * state);
const int32_t * gliner_get_token_ids(const struct gliner_state * state);
const int32_t * gliner_get_label_positions(const struct gliner_state * state); // n_scores entries
const float * gliner_get_label_states(const struct gliner_state * state); // n_scores * hidden_size

// Results belong to `state` and remain valid until its next score call or free.
// A failed score call clears previous results. NULL state returns 0/NULL.
int gliner_n_scores(const struct gliner_state * state);
const struct gliner_score * gliner_get_scores(const struct gliner_state * state);

#ifdef __cplusplus
}
#endif

#endif // GLINER_H
