#include "internal.h"

#include <utf8proc.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace gliner {
namespace {

std::vector<int32_t> codepoints(const std::string & text) {
    std::vector<int32_t> result;
    for (size_t i = 0; i < text.size();) {
        int32_t cp;
        const auto n = utf8proc_iterate(reinterpret_cast<const uint8_t *>(text.data() + i),
                                       static_cast<utf8proc_ssize_t>(text.size() - i), &cp);
        if (n <= 0 || cp == 0) throw std::invalid_argument("Input must be valid UTF-8 without NUL");
        result.push_back(cp);
        i += static_cast<size_t>(n);
    }
    return result;
}

void append(std::string & text, int32_t cp) {
    uint8_t bytes[4];
    const auto n = utf8proc_encode_char(cp, bytes);
    text.append(reinterpret_cast<const char *>(bytes), static_cast<size_t>(n));
}

// Rust regex's Unicode White_Space excludes Python's four information separators.
bool normalizer_space(int32_t cp) {
    return (cp >= 9 && cp <= 13) || cp == 32 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

} // namespace

Tokenizer::Tokenizer(const GgufFile & file) {
    if (file.string("tokenizer.gliner.pipeline") != "decide-nfc-v1" ||
        file.string("tokenizer.ggml.model") != "unigram") {
        throw std::runtime_error("Unsupported tokenizer pipeline");
    }
    const auto tokens = file.strings("tokenizer.ggml.tokens");
    scores_ = file.doubles("tokenizer.gliner.scores");
    unknown_ = file.u32("tokenizer.gliner.unknown_id");
    added_ = file.strings("tokenizer.gliner.added_tokens");
    added_ids_ = file.integers("tokenizer.gliner.added_ids");
    ranges_ = file.integers("tokenizer.gliner.unicode_ranges");
    const auto lower_from = file.integers("tokenizer.gliner.lower_from");
    const auto lower_to = file.strings("tokenizer.gliner.lower_to");
    if (tokens.size() != static_cast<size_t>(file.u32("deberta.vocab_size")) ||
        scores_.empty() || scores_.size() > tokens.size() ||
        static_cast<size_t>(unknown_) >= scores_.size() || added_.size() != added_ids_.size() ||
        lower_from.size() != lower_to.size() || ranges_.size() % 3) {
        throw std::runtime_error("Invalid tokenizer metadata sizes");
    }
    trie_.emplace_back();
    std::unordered_set<std::string> unique;
    for (size_t i = 0; i < scores_.size(); ++i) {
        if (tokens[i].empty() || !unique.insert(tokens[i]).second || !std::isfinite(scores_[i])) {
            throw std::runtime_error("Invalid Unigram vocabulary");
        }
        codepoints(tokens[i]);
        size_t node = 0;
        for (unsigned char byte : tokens[i]) {
            const auto found = trie_[node].children.find(byte);
            if (found != trie_[node].children.end()) node = found->second;
            else {
                const size_t next = trie_.size();
                trie_[node].children.emplace(byte, next);
                trie_.emplace_back();
                node = next;
            }
        }
        trie_[node].id = static_cast<int32_t>(i);
    }
    unknown_score_ = *std::min_element(scores_.begin(), scores_.end()) - 10.0;
    unique.clear();
    for (size_t i = 0; i < added_.size(); ++i) {
        if (added_[i].empty() || added_ids_[i] >= tokens.size() ||
            tokens[added_ids_[i]] != added_[i] || !unique.insert(added_[i]).second) {
            throw std::runtime_error("Invalid added token");
        }
    }
    for (const char * marker : {"[P]", "[L]", "[SEP_TEXT]", "[DESCRIPTION]"}) {
        if (!unique.count(marker)) throw std::runtime_error(std::string("Missing special token: ") + marker);
    }
    for (size_t i = 0; i < ranges_.size(); i += 3) {
        if (ranges_[i] > ranges_[i + 1] || ranges_[i + 1] > 0x10ffff ||
            ranges_[i + 2] > 15 || (i && ranges_[i] <= ranges_[i - 2])) {
            throw std::runtime_error("Invalid Unicode property ranges");
        }
    }
    for (size_t i = 0; i < lower_from.size(); ++i) {
        if (lower_from[i] > 0x10ffff || lower_to[i].empty() ||
            !lower_.emplace(lower_from[i], lower_to[i]).second) {
            throw std::runtime_error("Invalid Unicode lowercase mapping");
        }
        codepoints(lower_to[i]);
    }
}

unsigned Tokenizer::flags(int32_t cp) const {
    size_t lo = 0, hi = ranges_.size() / 3;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (ranges_[3 * mid + 1] < static_cast<uint32_t>(cp)) lo = mid + 1;
        else hi = mid;
    }
    return lo < ranges_.size() / 3 && ranges_[3 * lo] <= static_cast<uint32_t>(cp) ? ranges_[3 * lo + 2] : 0;
}

std::string Tokenizer::lowercase(const std::vector<int32_t> & word) const {
    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        const int32_t cp = word[i];
        if (cp == 0x3a3) {
            size_t before = i, after = i + 1;
            while (before && (flags(word[before - 1]) & 8)) --before;
            while (after < word.size() && (flags(word[after]) & 8)) ++after;
            const bool final = before && (flags(word[before - 1]) & 4) &&
                               (after == word.size() || !(flags(word[after]) & 4));
            append(result, final ? 0x3c2 : 0x3c3);
        } else {
            const auto found = lower_.find(cp);
            if (found == lower_.end()) append(result, cp);
            else result += found->second;
        }
    }
    return result;
}

std::vector<Word> Tokenizer::words(const std::string & text, int max_words) const {
    auto chars = codepoints(text);
    std::vector<size_t> offsets{0};
    for (int32_t cp : chars) {
        uint8_t bytes[4];
        offsets.push_back(offsets.back() + static_cast<size_t>(utf8proc_encode_char(cp, bytes)));
    }
    if (chars.empty() || (chars.back() != '.' && chars.back() != '!' && chars.back() != '?')) chars.push_back('.');
    if (offsets.size() == chars.size()) offsets.push_back(text.size());
    std::string shadow;
    for (int32_t cp : chars) {
        if (flags(cp) & 2) shadow += ' ';
        else if (cp == 0x130 || cp == 0x131) shadow += 'i';
        else if (cp == 0x17f) shadow += 's';
        else if (cp == 0x212a) shadow += 'k';
        else if (cp >= 'A' && cp <= 'Z') shadow += static_cast<char>(cp + 32);
        else shadow += cp < 128 ? static_cast<char>(cp) : '\x7f';
    }
    auto letter = [](char c) { return c >= 'a' && c <= 'z'; };
    auto digit = [](char c) { return c >= '0' && c <= '9'; };
    auto alnum = [&](char c) { return letter(c) || digit(c); };
    std::vector<Word> result;
    for (size_t i = 0; i < chars.size() && (!max_words || result.size() < static_cast<size_t>(max_words));) {
        if (flags(chars[i]) & 2) { ++i; continue; }
        size_t special_end = i;
        for (const std::string prefix : {"https://", "http://", "www."}) {
            if (shadow.compare(i, prefix.size(), prefix) == 0 && i + prefix.size() < shadow.size() &&
                shadow[i + prefix.size()] != ' ') {
                special_end = i + prefix.size();
                while (special_end < shadow.size() && shadow[special_end] != ' ') ++special_end;
                break;
            }
        }
        if (special_end == i) {
            size_t at = i;
            while (at < shadow.size() && (alnum(shadow[at]) || std::string("._%+-").find(shadow[at]) != std::string::npos)) ++at;
            if (at > i && at < shadow.size() && shadow[at] == '@') {
                int suffix_letters = -1;
                for (size_t j = at + 1; j < shadow.size() && (alnum(shadow[j]) || shadow[j] == '.' || shadow[j] == '-'); ++j) {
                    if (shadow[j] == '.' && j > at + 1) suffix_letters = 0;
                    else if (letter(shadow[j]) && suffix_letters >= 0) ++suffix_letters;
                    else suffix_letters = -1;
                    if (suffix_letters >= 2) special_end = j + 1;
                }
            }
        }
        if (special_end == i && shadow[i] == '@') {
            size_t end = i + 1;
            while (end < shadow.size() && (alnum(shadow[end]) || shadow[end] == '_')) ++end;
            if (end > i + 1) special_end = end;
        }
        size_t end = special_end > i ? special_end : i + 1;
        if (special_end == i && (flags(chars[i]) & 1)) {
            while (end < chars.size()) {
                if (flags(chars[end]) & 1) ++end;
                else if (chars[end] == '-' && end + 1 < chars.size() && (flags(chars[end + 1]) & 1)) end += 2;
                else break;
            }
        }
        result.push_back({lowercase(std::vector<int32_t>(chars.begin() + i, chars.begin() + end)), offsets[i], offsets[end]});
        i = end;
    }
    return result;
}

std::string Tokenizer::normalize(const std::string & text) const {
    const auto chars = codepoints(text);
    std::string replaced;
    for (size_t i = 0; i < chars.size();) {
        if (normalizer_space(chars[i])) {
            size_t end = i + 1;
            while (end < chars.size() && normalizer_space(chars[end])) ++end;
            if (end - i >= 2 || chars[i] == '\n' || chars[i] == '\r' || chars[i] == '\t') replaced += ' ';
            else append(replaced, chars[i]);
            i = end;
        } else append(replaced, chars[i++]);
    }
    uint8_t * output = nullptr;
    const auto n = utf8proc_map(reinterpret_cast<const uint8_t *>(replaced.data()),
                               static_cast<utf8proc_ssize_t>(replaced.size()), &output, UTF8PROC_COMPOSE);
    if (n < 0) throw std::runtime_error(utf8proc_errmsg(n));
    std::unique_ptr<uint8_t, decltype(&std::free)> owner(output, std::free);
    auto normalized = codepoints(std::string(reinterpret_cast<char *>(output), static_cast<size_t>(n)));
    while (!normalized.empty() && normalizer_space(normalized.back())) normalized.pop_back();
    std::string result;
    for (int32_t cp : normalized) append(result, cp);
    return result;
}

void Tokenizer::unigram(const std::string & text, std::vector<int32_t> & ids) const {
    struct Path { double score = 0; size_t from = 0; int32_t id = -1; };
    std::vector<Path> best(text.size() + 1);
    for (size_t begin = 0; begin < text.size();) {
        int32_t cp;
        const size_t width = static_cast<size_t>(utf8proc_iterate(
            reinterpret_cast<const uint8_t *>(text.data() + begin), text.size() - begin, &cp));
        auto update = [&](size_t end, int32_t id, double score) {
            score += best[begin].score;
            if (best[end].id < 0 || score > best[end].score) best[end] = {score, begin, id};
        };
        size_t node = 0;
        bool single = false;
        for (size_t end = begin; end < text.size(); ++end) {
            const auto found = trie_[node].children.find(static_cast<unsigned char>(text[end]));
            if (found == trie_[node].children.end()) break;
            node = found->second;
            const auto id = trie_[node].id;
            if (id >= 0) {
                update(end + 1, id, scores_[id]);
                if (end + 1 - begin == width) single = true;
            }
        }
        if (!single) update(begin + width, unknown_, unknown_score_);
        begin += width;
    }
    std::vector<int32_t> reversed;
    for (size_t end = text.size(); end;) {
        const auto & node = best[end];
        if (node.id < 0 || node.from >= end) throw std::runtime_error("Unigram path construction failed");
        if (node.id != unknown_ || reversed.empty() || reversed.back() != unknown_) reversed.push_back(node.id);
        end = node.from;
    }
    ids.insert(ids.end(), reversed.rbegin(), reversed.rend());
}

void Tokenizer::segment(const std::string & text, std::vector<int32_t> & ids) const {
    codepoints(text);
    size_t start = 0;
    while (start < text.size()) {
        size_t at = std::string::npos, which = 0;
        for (size_t i = 0; i < added_.size(); ++i) {
            const size_t found = text.find(added_[i], start);
            if (found < at || (found == at && found != std::string::npos && added_[i].size() > added_[which].size())) {
                at = found;
                which = i;
            }
        }
        const size_t end = at == std::string::npos ? text.size() : at;
        auto normal = normalize(text.substr(start, end - start));
        if (!normal.empty()) {
            std::string piece;
            for (int32_t cp : codepoints(normal)) {
                if (cp == ' ' || cp == 0x2581) {
                    if (!piece.empty()) unigram(piece, ids);
                    piece = "\xe2\x96\x81";
                } else {
                    if (piece.empty()) piece = "\xe2\x96\x81";
                    append(piece, cp);
                }
            }
            if (!piece.empty()) unigram(piece, ids);
        }
        if (at == std::string::npos) break;
        ids.push_back(static_cast<int32_t>(added_ids_[which]));
        start = at + added_[which].size();
    }
}

Tokenized Tokenizer::encode(const std::string & text, const std::vector<ClassificationTask> & tasks,
                           int max_words, int max_tokens, bool entities) const {
    if (tasks.empty()) throw std::invalid_argument("Classification tasks are required");
    if (entities && std::find(added_.begin(), added_.end(), "[E]") == added_.end()) {
        throw std::runtime_error("Missing [E] token required for span extraction");
    }
    if (tasks.size() > 1 && std::find(added_.begin(), added_.end(), "[SEP_STRUCT]") == added_.end()) {
        throw std::runtime_error("Missing [SEP_STRUCT] token required for joint classification");
    }
    Tokenized result;
    auto emit = [&](const std::string & value) {
        segment(value, result.ids);
        if (result.ids.size() > static_cast<size_t>(max_tokens)) {
            throw std::invalid_argument("Token limit exceeded; use max_words to truncate text or increase max_tokens");
        }
    };
    std::unordered_set<std::string> task_names;
    for (const auto & task : tasks) {
        if (task.name.empty() || !task_names.insert(task.name).second) {
            throw std::invalid_argument("Task names must be nonempty and unique");
        }
        if (task.labels.empty()) throw std::invalid_argument("Each task requires labels");
        std::string prompt_text = task.prompt.empty() ? task.name : task.name + ": " + task.prompt;
        std::unordered_set<std::string> seen;
        for (size_t i = 0; i < task.labels.size(); ++i) {
            const auto & label = task.labels[i];
            if (label.empty() || !seen.insert(label).second) throw std::invalid_argument("Labels must be nonempty and unique within each task");
            if (label == "[SEP_TEXT]" || label == "[SEP_STRUCT]") {
                throw std::invalid_argument("Schema separators cannot be used as label names");
            }
            if (!task.descriptions.empty() && task.descriptions[i]) {
                prompt_text += " [DESCRIPTION] " + label + ": " + *task.descriptions[i];
            }
        }
        if (prompt_text == "[SEP_TEXT]" || prompt_text == "[SEP_STRUCT]") {
            throw std::invalid_argument("Schema separators cannot be used as task names");
        }
        if (!result.tasks.empty()) emit("[SEP_STRUCT]");
        result.tasks.push_back({static_cast<int>(result.markers.size()), static_cast<int>(task.labels.size())});
        emit("(");
        result.prompt_position = static_cast<int32_t>(result.ids.size());
        emit("[P]"); emit(prompt_text); emit("(");
        for (const auto & label : task.labels) {
            result.markers.push_back(static_cast<int32_t>(result.ids.size()));
            emit(entities ? "[E]" : "[L]");
            emit(label);
        }
        emit(")"); emit(")");
    }
    emit("[SEP_TEXT]");
    for (const auto & word : words(text, max_words)) {
        const auto position = static_cast<int32_t>(result.ids.size());
        emit(word.text);
        if (entities) {
            if (result.ids.size() == static_cast<size_t>(position)) {
                throw std::invalid_argument("Text word produced no subwords; span alignment would be ambiguous");
            }
            result.word_positions.push_back(position);
            result.words.push_back(word);
        }
    }
    return result;
}

} // namespace gliner
