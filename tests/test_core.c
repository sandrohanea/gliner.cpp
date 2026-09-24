#include "gliner/gliner.h"

#include <math.h>
#include <stdio.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "check failed: %s at line %d\n", #expr, __LINE__); \
    return 1; \
} } while (0)

int main(int argc, char ** argv) {
    CHECK(gliner_model_hidden_size(NULL) == 0);
    CHECK(gliner_model_n_tensors(NULL) == 0);
    CHECK(gliner_model_n_layers(NULL) == 0);
    CHECK(gliner_n_scores(NULL) == 0);
    CHECK(gliner_get_scores(NULL) == NULL);
    CHECK(gliner_init_from_file(NULL) == NULL);
    CHECK(gliner_last_error()[0] != '\0');
    CHECK(gliner_init_state(NULL) == NULL);
    CHECK(gliner_last_error()[0] != '\0');
    gliner_free(NULL);
    gliner_free_state(NULL);
    if (argc == 1) return 0;

    struct gliner_context * ctx = gliner_init_from_file(argv[1]);
    CHECK(ctx != NULL);
    CHECK(gliner_model_hidden_size(ctx) == 2);
    CHECK(gliner_model_n_tensors(ctx) == 5);
    CHECK(gliner_model_n_layers(ctx) == 1);
    struct gliner_state * state = gliner_init_state(ctx);
    CHECK(state != NULL);
    struct gliner_state * state2 = gliner_init_state(ctx);
    CHECK(state2 != NULL);

    const float labels[4] = {1.f, 2.f, -1.f, 2.f};
    CHECK(gliner_score_label_states(ctx, state, labels, 2, 2) == GLINER_STATUS_OK);
    CHECK(gliner_n_scores(state) == 2);
    const struct gliner_score * scores = gliner_get_scores(state);
    CHECK(scores != NULL);
    CHECK(fabsf(scores[0].logit - 5.5f) < 1e-5f);
    CHECK(fabsf(scores[1].logit - 3.5f) < 1e-5f);
    CHECK(gliner_score_label_states(ctx, state2, labels + 2, 1, 1) == GLINER_STATUS_OK);
    CHECK(gliner_n_scores(state2) == 1);
    CHECK(fabsf(gliner_get_scores(state2)[0].logit - 3.5f) < 1e-5f);
    CHECK(gliner_n_scores(state) == 2);

    CHECK(gliner_score_label_states(ctx, state, labels, 1, 1) == GLINER_STATUS_OK);
    CHECK(gliner_n_scores(state) == 1);
    CHECK(gliner_score_label_states(ctx, state, NULL, 1, 1) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_n_scores(state) == 0);
    CHECK(gliner_get_scores(state) == NULL);
    CHECK(gliner_last_error()[0] != '\0');
    CHECK(gliner_n_scores(state2) == 1);

    gliner_free_state(state2);
    gliner_free_state(state);
    gliner_free(ctx);
    return 0;
}
