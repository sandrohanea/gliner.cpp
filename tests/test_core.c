#include "gliner/gliner.h"

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "check failed: %s at line %d\n", #expr, __LINE__); \
    return 1; \
} } while (0)

struct memory_stream {
    const unsigned char * data;
    size_t size;
    size_t position;
    size_t chunk;
    size_t fail_at;
    int overread;
    int closes;
    size_t largest_request;
};

static size_t stream_read(void * context, void * output, size_t count) {
    struct memory_stream * stream = (struct memory_stream *)context;
    if (stream->closes || count > 8 * 1024 * 1024) return count + 1;
    if (count > stream->largest_request) stream->largest_request = count;
    if (stream->overread) return count + 1;
    if (stream->position >= stream->fail_at) return 0;
    if (count > stream->size - stream->position) count = stream->size - stream->position;
    if (count > stream->chunk) count = stream->chunk;
    if (count > stream->fail_at - stream->position) count = stream->fail_at - stream->position;
    memcpy(output, stream->data + stream->position, count);
    stream->position += count;
    return count;
}

static bool stream_eof(void * context) {
    struct memory_stream * stream = (struct memory_stream *)context;
    return stream->position == stream->size;
}

static void stream_close(void * context) {
    ((struct memory_stream *)context)->closes++;
}

static void span_layer(int layer, const float * values, int tokens, int hidden, void * context) {
    int * count = (int *)context;
    if (layer != *count || !values || tokens <= 0 || hidden <= 0) *count = -1000;
    else ++*count;
}

static int test_spans(const char * path) {
    struct gliner_context * ctx = gliner_init_from_file(path);
    CHECK(ctx != NULL && gliner_model_supports_spans(ctx));
    struct gliner_state * state = gliner_init_state(ctx);
    CHECK(state != NULL);
    const struct gliner_span_label labels[] = {{"person", "A person name"}, {"thing", NULL}};
    const char * text = "Caf\xc3\xa9 Tim Cook works at Apple.";
    struct gliner_span_params params = gliner_default_span_params();
    params.threshold = 0;
    params.max_spans_per_label = 1;
    int calls = 0;
    gliner_set_eval_callback(state, span_layer, &calls);
    CHECK(gliner_extract_spans(ctx, state, text, labels, 2, &params) == GLINER_STATUS_OK);
    CHECK(calls == gliner_model_n_layers(ctx) + 1);
    gliner_set_eval_callback(state, NULL, NULL);
    CHECK(gliner_n_spans(state) == 2);
    CHECK(gliner_n_scores(state) == 0 && gliner_n_tasks(state) == 0);
    CHECK(gliner_get_label_positions(state) == NULL && gliner_get_label_states(state) == NULL);
    const struct gliner_span * result = gliner_get_spans(state);
    CHECK(result != NULL);
    for (int i = 0; i < 2; ++i) {
        CHECK(result[i].label_index == i);
        CHECK(result[i].start < result[i].end && result[i].end <= strlen(text));
        CHECK(strlen(result[i].text) == result[i].end - result[i].start);
        CHECK(memcmp(result[i].text, text + result[i].start, result[i].end - result[i].start) == 0);
        CHECK(isfinite(result[i].logit) && result[i].probability >= 0 && result[i].probability <= 1);
    }
    const struct gliner_span_scores * raw = gliner_get_span_scores(state);
    CHECK(raw && raw->n_labels == 2 && raw->predicted_count == 1 && raw->n_words > 0);
    CHECK(raw->n_candidates > 0 && raw->max_width == 8);
    CHECK(raw->label_positions[1] > raw->label_positions[0]);
    for (int i = 0; i < raw->n_candidates; ++i) {
        CHECK(raw->start_words[i] < raw->end_words[i] && raw->end_words[i] <= raw->n_words);
        CHECK(raw->end_words[i] - raw->start_words[i] <= raw->max_width);
    }
    params.max_spans_per_label = 0;
    params.allow_overlap = 1;
    CHECK(gliner_extract_spans(ctx, state, text, labels, 2, &params) == GLINER_STATUS_OK);
    CHECK(gliner_n_spans(state) > 2);
    CHECK(gliner_extract_spans(ctx, state, "", labels, 2, &params) == GLINER_STATUS_OK);
    CHECK(gliner_n_spans(state) == 0 && gliner_get_spans(state) == NULL);
    CHECK(gliner_get_span_scores(state) != NULL);
    params.max_tokens = 2;
    CHECK(gliner_extract_spans(ctx, state, text, labels, 2, &params) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_get_span_scores(state) == NULL && gliner_n_tokens(state) == 0);
    params = gliner_default_span_params();
    params.threshold = NAN;
    CHECK(gliner_extract_spans(ctx, state, text, labels, 2, &params) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_extract_spans(ctx, state, text, NULL, 2, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_extract_spans(ctx, state, text, labels, INT_MAX, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_extract_spans(ctx, state, "\xff", labels, 2, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    const struct gliner_span_label duplicates[] = {{"same", NULL}, {"same", NULL}};
    CHECK(gliner_extract_spans(ctx, state, text, duplicates, 2, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_extract_spans(ctx, state, text, labels, 2, NULL) == GLINER_STATUS_OK);
    const char * classes[] = {"yes", "no"};
    CHECK(gliner_classify_text(ctx, state, text, "intent", classes, 2, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_n_spans(state) == 0 && gliner_get_spans(state) == NULL && gliner_get_span_scores(state) == NULL);
    CHECK(gliner_extract_spans(ctx, state, text, labels, 2, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_score_label_states(ctx, state, NULL, 1, 1) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_n_spans(state) == 0 && gliner_get_span_scores(state) == NULL);
    gliner_free_state(state);
    gliner_free(ctx);
    return 0;
}

struct eval_count {
    int layers;
    int tokens;
    int invalid;
};

static void count_layer(int layer, const float * values, int tokens, int hidden, void * context) {
    struct eval_count * count = (struct eval_count *)context;
    if (layer != count->layers || values == NULL || tokens <= 0 || hidden != 2) count->invalid = 1;
    if (layer > 0 && tokens != count->tokens) count->invalid = 1;
    count->tokens = tokens;
    count->layers++;
}

static int test_batch(struct gliner_context * ctx) {
    struct gliner_state * state = gliner_init_state(ctx);
    struct gliner_state * separate = gliner_init_state(ctx);
    CHECK(state != NULL && separate != NULL);
    const char * first[] = {"yes", "no"};
    const char * second[] = {"yes", "no", "unknown"};
    const char * descriptions[] = {"", "not applicable"};
    struct gliner_classification_task tasks[] = {
        {"intent", first, 2, "Choose [L] carefully", descriptions},
        {"intent detail", second, 3, NULL, NULL},
        {"topic", first, 1, NULL, NULL}
    };
    struct gliner_batch_params params = gliner_default_batch_params();
    struct gliner_text_params single = gliner_default_text_params();
    CHECK(params.n_threads == single.n_threads && params.max_tokens == single.max_tokens && params.max_words == single.max_words);
    single.prompt = tasks[0].prompt;
    single.label_descriptions = descriptions;
    CHECK(gliner_classify_text(ctx, separate, "hello world", tasks[0].task, first, 2, &single) == GLINER_STATUS_OK);
    CHECK(gliner_classify_text_batch(ctx, state, "hello world", tasks, 1, &params) == GLINER_STATUS_OK);
    CHECK(gliner_n_tasks(state) == 1 && gliner_n_tasks(separate) == 1);
    CHECK(gliner_n_tokens(state) == gliner_n_tokens(separate));
    CHECK(memcmp(gliner_get_token_ids(state), gliner_get_token_ids(separate), (size_t)gliner_n_tokens(state) * sizeof(int32_t)) == 0);
    CHECK(memcmp(gliner_get_scores(state), gliner_get_scores(separate), 2 * sizeof(struct gliner_score)) == 0);

    struct eval_count count = {0, 0, 0};
    gliner_set_eval_callback(state, count_layer, &count);
    CHECK(gliner_classify_text_batch(ctx, state, "hello world", tasks, 3, &params) == GLINER_STATUS_OK);
    CHECK(count.layers == gliner_model_n_layers(ctx) + 1 && !count.invalid);
    CHECK(count.tokens == gliner_n_tokens(state));
    CHECK(gliner_n_tasks(state) == 3 && gliner_n_scores(state) == 6);
    const struct gliner_task_result * ranges = gliner_get_task_results(state);
    CHECK(ranges != NULL);
    CHECK(ranges[0].score_offset == 0 && ranges[0].n_scores == 2);
    CHECK(ranges[1].score_offset == 2 && ranges[1].n_scores == 3);
    CHECK(ranges[2].score_offset == 5 && ranges[2].n_scores == 1);
    const int32_t * positions = gliner_get_label_positions(state);
    for (int i = 0; i < 6; ++i) {
        CHECK(isfinite(gliner_get_scores(state)[i].logit));
        CHECK(positions[i] >= 0 && positions[i] < gliner_n_tokens(state));
        if (i) CHECK(positions[i] > positions[i - 1]);
    }
    struct gliner_score saved[6];
    memcpy(saved, gliner_get_scores(state), sizeof(saved));
    gliner_set_eval_callback(state, NULL, NULL);
    CHECK(gliner_classify_text_batch(ctx, state, "hello world", tasks, 3, NULL) == GLINER_STATUS_OK);
    CHECK(memcmp(saved, gliner_get_scores(state), sizeof(saved)) == 0);
    CHECK(gliner_n_tasks(separate) == 1 && gliner_n_scores(separate) == 2);

    params.max_tokens = 10;
    CHECK(gliner_classify_text_batch(ctx, state, "hello world", tasks, 3, &params) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_n_tasks(state) == 0 && gliner_get_task_results(state) == NULL);
    CHECK(gliner_n_scores(state) == 0 && gliner_n_tokens(state) == 0);
    CHECK(gliner_get_label_states(state) == NULL);
    params = gliner_default_batch_params();
    params.max_words = 1;
    CHECK(gliner_classify_text_batch(ctx, state, "hello world", tasks, 3, &params) == GLINER_STATUS_OK);
    CHECK(gliner_classify_text_batch(ctx, state, "", tasks, 3, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_classify_text_batch(ctx, state, "x", NULL, 1, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 0, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, INT_MAX, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_classify_text_batch(ctx, state, NULL, tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_classify_text_batch(ctx, state, "\xff", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    tasks[1].task = tasks[0].task;
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    tasks[1].task = "other";
    tasks[1].n_labels = INT_MAX;
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    tasks[1].n_labels = 0;
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    tasks[1].n_labels = 3;
    tasks[1].labels = NULL;
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    const char * null_label[] = {"yes", NULL, "unknown"};
    tasks[1].labels = null_label;
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    tasks[1].labels = second;
    tasks[1].task = NULL;
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    tasks[1].task = "";
    CHECK(gliner_classify_text_batch(ctx, state, "x", tasks, 3, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    tasks[1].task = "other";
    CHECK(gliner_classify_text_batch(ctx, state, "hello world", tasks, 3, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_classify_text(ctx, state, "hello world", "intent", first, 2, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_n_tasks(state) == 1 && gliner_get_task_results(state)[0].n_scores == 2);
    const float states[] = {1, 2};
    CHECK(gliner_score_label_states(ctx, state, states, 1, 1) == GLINER_STATUS_OK);
    CHECK(gliner_n_tasks(state) == 0 && gliner_get_task_results(state) == NULL);
    gliner_free_state(separate);
    gliner_free_state(state);
    return 0;
}

static int compare_models(struct gliner_context * a, struct gliner_context * b) {
    CHECK(gliner_model_hidden_size(a) == gliner_model_hidden_size(b));
    CHECK(gliner_model_n_tensors(a) == gliner_model_n_tensors(b));
    CHECK(strcmp(gliner_model_backend_name(a), gliner_model_backend_name(b)) == 0);
    struct gliner_state * sa = gliner_init_state(a);
    struct gliner_state * sb = gliner_init_state(b);
    CHECK(sa != NULL && sb != NULL);
    const char * labels[] = {"first", "second"};
    CHECK(gliner_classify_text(a, sa, "hello world", "intent", labels, 2, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_classify_text(b, sb, "hello world", "intent", labels, 2, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_n_tokens(sa) == gliner_n_tokens(sb));
    CHECK(memcmp(gliner_get_token_ids(sa), gliner_get_token_ids(sb),
                 (size_t)gliner_n_tokens(sa) * sizeof(int32_t)) == 0);
    CHECK(memcmp(gliner_get_label_positions(sa), gliner_get_label_positions(sb), 2 * sizeof(int32_t)) == 0);
    for (int i = 0; i < 4; ++i) CHECK(fabsf(gliner_get_label_states(sa)[i] - gliner_get_label_states(sb)[i]) < 1e-5f);
    for (int i = 0; i < 2; ++i) CHECK(fabsf(gliner_get_scores(sa)[i].logit - gliner_get_scores(sb)[i].logit) < 1e-5f);
    const struct gliner_classification_task tasks[] = {
        {"intent", labels, 2, NULL, NULL},
        {"topic", labels, 1, NULL, NULL}
    };
    CHECK(gliner_classify_text_batch(a, sa, "hello world", tasks, 2, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_classify_text_batch(b, sb, "hello world", tasks, 2, NULL) == GLINER_STATUS_OK);
    CHECK(gliner_n_tasks(sa) == 2 && gliner_n_tasks(sb) == 2);
    CHECK(gliner_n_scores(sa) == 3 && gliner_n_scores(sb) == 3);
    CHECK(gliner_n_tokens(sa) == gliner_n_tokens(sb));
    CHECK(memcmp(gliner_get_token_ids(sa), gliner_get_token_ids(sb),
                 (size_t)gliner_n_tokens(sa) * sizeof(int32_t)) == 0);
    for (int i = 0; i < 3; ++i) CHECK(fabsf(gliner_get_scores(sa)[i].logit - gliner_get_scores(sb)[i].logit) < 1e-5f);
    gliner_free_state(sa);
    gliner_free_state(sb);
    return 0;
}

static int test_loading(const char * path, struct gliner_context * reference) {
    FILE * file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    const long length = ftell(file);
    CHECK(length > 64 && fseek(file, 0, SEEK_SET) == 0);
    const size_t size = (size_t)length;
    unsigned char * bytes = (unsigned char *)malloc(size);
    CHECK(bytes != NULL);
    CHECK(fread(bytes, 1, size, file) == size);
    CHECK(fclose(file) == 0);

    const size_t truncated[] = {1, 23, size / 2, size - 32};
    for (size_t i = 0; i < sizeof(truncated) / sizeof(truncated[0]); ++i) {
        CHECK(gliner_init_from_buffer(bytes, truncated[i]) == NULL);
        CHECK(gliner_last_error()[0] != '\0');
        struct memory_stream stream = {bytes, truncated[i], 0, 7, SIZE_MAX, 0, 0};
        struct gliner_model_loader loader = {&stream, stream_read, stream_eof, stream_close};
        CHECK(gliner_init(&loader) == NULL);
        CHECK(gliner_last_error()[0] != '\0');
        CHECK(stream.closes == 1);
    }
    bytes[0] = 'X';
    CHECK(gliner_init_from_buffer(bytes, size) == NULL);
    struct memory_stream stream = {bytes, size, 0, 7, SIZE_MAX, 0, 0};
    struct gliner_model_loader loader = {&stream, stream_read, stream_eof, stream_close};
    CHECK(gliner_init(&loader) == NULL);
    CHECK(stream.closes == 1);
    bytes[0] = 'G';

    stream = (struct memory_stream){bytes, size, 0, 7, 24, 0, 0};
    CHECK(gliner_init(&loader) == NULL);
    CHECK(strstr(gliner_last_error(), "no progress") != NULL);
    CHECK(stream.closes == 1);
    stream = (struct memory_stream){bytes, size, 0, 7, size - 32, 0, 0};
    CHECK(gliner_init(&loader) == NULL);
    CHECK(strstr(gliner_last_error(), "no progress") != NULL);
    CHECK(stream.closes == 1);
    stream = (struct memory_stream){bytes, size, 0, 7, SIZE_MAX, 1, 0};
    CHECK(gliner_init(&loader) == NULL);
    CHECK(strstr(gliner_last_error(), "more bytes") != NULL);
    CHECK(stream.closes == 1);
    stream.closes = 0;
    struct gliner_model_loader invalid = loader;
    invalid.read = NULL;
    CHECK(gliner_init(&invalid) == NULL && stream.closes == 0);
    invalid = loader; invalid.eof = NULL;
    CHECK(gliner_init(&invalid) == NULL && stream.closes == 0);
    invalid = loader; invalid.close = NULL;
    CHECK(gliner_init(&invalid) == NULL && stream.closes == 0);

    struct gliner_context * buffer_model = gliner_init_from_buffer(bytes, size);
    CHECK(buffer_model != NULL);
    unsigned char * prefixed = (unsigned char *)malloc(size + 17);
    CHECK(prefixed != NULL);
    memset(prefixed, 0xa5, 17);
    memcpy(prefixed + 17, bytes, size);
    stream = (struct memory_stream){prefixed, size + 17, 17, 7, SIZE_MAX, 0, 0};
    struct gliner_context * stream_model = gliner_init(&loader);
    CHECK(stream_model != NULL && stream.closes == 1);
    if (size > 8 * 1024 * 1024) CHECK(stream.largest_request == 8 * 1024 * 1024);
    memset(bytes, 0, size);
    memset(prefixed, 0, size + 17);
    free(bytes);
    free(prefixed);
    CHECK(compare_models(reference, buffer_model) == 0);
    CHECK(compare_models(reference, stream_model) == 0);
    gliner_free(buffer_model);
    gliner_free(stream_model);
    CHECK(stream.closes == 1);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--spans") == 0) return test_spans(argv[2]);
    CHECK(gliner_model_hidden_size(NULL) == 0);
    CHECK(gliner_model_n_tensors(NULL) == 0);
    CHECK(gliner_model_n_layers(NULL) == 0);
    CHECK(gliner_model_supports_text(NULL) == 0);
    CHECK(gliner_model_supports_spans(NULL) == 0);
    CHECK(gliner_n_spans(NULL) == 0);
    CHECK(gliner_get_spans(NULL) == NULL);
    CHECK(gliner_get_span_scores(NULL) == NULL);
    CHECK(gliner_model_backend_name(NULL) == NULL);
    CHECK(gliner_model_device_name(NULL) == NULL);
    CHECK(gliner_n_scores(NULL) == 0);
    CHECK(gliner_n_tasks(NULL) == 0);
    CHECK(gliner_get_task_results(NULL) == NULL);
    CHECK(gliner_n_tokens(NULL) == 0);
    CHECK(gliner_get_token_ids(NULL) == NULL);
    CHECK(gliner_get_label_positions(NULL) == NULL);
    CHECK(gliner_get_label_states(NULL) == NULL);
    CHECK(gliner_get_scores(NULL) == NULL);
    CHECK(gliner_init_from_file(NULL) == NULL);
    CHECK(gliner_last_error()[0] != '\0');
    CHECK(gliner_init_state(NULL) == NULL);
    CHECK(gliner_last_error()[0] != '\0');
    CHECK(gliner_init_from_buffer(NULL, 16) == NULL);
    CHECK(gliner_init_from_buffer("GGUF", 0) == NULL);
    CHECK(gliner_init(NULL) == NULL);
    CHECK(gliner_last_error()[0] != '\0');
    gliner_free(NULL);
    gliner_free_state(NULL);
    CHECK(gliner_build_backend() != NULL);
    if (argc == 1) return 0;

    const char * expected_backend = argc >= 3 ? argv[2] : gliner_build_backend();
    CHECK(strcmp(gliner_build_backend(), expected_backend) == 0);
    struct gliner_context * ctx = gliner_init_from_file(argv[1]);
    if (!ctx) fprintf(stderr, "%s\n", gliner_last_error());
    CHECK(ctx != NULL);
    CHECK(strcmp(gliner_model_backend_name(ctx), expected_backend) == 0);
    CHECK(gliner_model_device_name(ctx) != NULL);
    CHECK(gliner_model_hidden_size(ctx) == 2);
    CHECK(gliner_model_n_tensors(ctx) == 27);
    CHECK(gliner_model_n_layers(ctx) == 1);
    CHECK(gliner_model_supports_text(ctx) == 1);
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

    const char * names[] = {"first", "second"};
    struct gliner_text_params params = gliner_default_text_params();
    params.n_threads = 2;
    CHECK(gliner_classify_text(ctx, state, "hello world", "intent", names, 2, &params) == GLINER_STATUS_OK);
    CHECK(gliner_n_scores(state) == 2);
    CHECK(gliner_n_tokens(state) > 0);
    CHECK(gliner_get_token_ids(state) != NULL);
    CHECK(gliner_get_label_positions(state) != NULL);
    CHECK(gliner_get_label_states(state) != NULL);
    const float first_logit = gliner_get_scores(state)[0].logit;
    CHECK(isfinite(first_logit));
    CHECK(gliner_classify_text(ctx, state2, "hello world", "intent", names, 2, NULL) == GLINER_STATUS_OK);
    CHECK(fabsf(gliner_get_scores(state2)[0].logit - first_logit) < 1e-4f);
    CHECK(gliner_classify_text(ctx, state, "", "intent", names, 2, NULL) == GLINER_STATUS_OK);
    params.max_tokens = 2;
    CHECK(gliner_classify_text(ctx, state, "hello", "intent", names, 2, &params) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_n_scores(state) == 0);
    CHECK(gliner_n_tokens(state) == 0);
    CHECK(gliner_get_label_states(state) == NULL);
    CHECK(gliner_n_scores(state2) == 2);
    CHECK(gliner_classify_text(ctx, state, "\xff", "intent", names, 2, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_classify_text(ctx, state, NULL, "intent", names, 2, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    const char * duplicates[] = {"first", "first"};
    CHECK(gliner_classify_text(ctx, state, "hello", "intent", duplicates, 2, NULL) == GLINER_STATUS_INVALID_ARGUMENT);
    const float invalid[2] = {NAN, 0.f};
    CHECK(gliner_score_label_states(ctx, state, invalid, 1, 1) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_score_label_states(ctx, state2, labels, 2, 1) == GLINER_STATUS_OK);
    CHECK(gliner_n_tokens(state2) == 0);
    CHECK(gliner_get_label_states(state2) == NULL);
    struct gliner_context * other = gliner_init_from_file(argv[1]);
    CHECK(other != NULL);
    CHECK(strcmp(gliner_model_backend_name(other), expected_backend) == 0);
    CHECK(gliner_score_label_states(other, state2, labels, 2, 1) == GLINER_STATUS_INVALID_ARGUMENT);
    CHECK(gliner_n_scores(state2) == 0);
    CHECK(gliner_classify_text(ctx, state2, "hello world", "intent", names, 2, NULL) == GLINER_STATUS_OK);
    CHECK(fabsf(gliner_get_scores(state2)[0].logit - first_logit) < 1e-4f);
    gliner_free(other);
    CHECK(test_loading(argv[1], ctx) == 0);
    CHECK(test_batch(ctx) == 0);

    gliner_free_state(state2);
    gliner_free_state(state);
    gliner_free(ctx);
    return 0;
}
