#ifndef GLINER_H
#define GLINER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The context owns GGUF metadata and immutable classifier weights. A state owns inference
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

// Score precomputed contextual [L] marker states. `label_states` is row-major
// [n_labels, hidden_size]. Full text-to-state inference is not implemented yet.
// Each state may be used by only one thread at a time; distinct states can run
// concurrently against the same context. Returns a gliner_status value.
int gliner_score_label_states(
    const struct gliner_context * ctx,
    struct gliner_state * state,
    const float * label_states,
    int n_labels,
    int n_threads);

// Results belong to `state` and remain valid until its next score call or free.
// A failed score call clears previous results. NULL state returns 0/NULL.
int gliner_n_scores(const struct gliner_state * state);
const struct gliner_score * gliner_get_scores(const struct gliner_state * state);

#ifdef __cplusplus
}
#endif

#endif // GLINER_H
