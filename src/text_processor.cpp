#include "text_processor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace paper_qa {

    TextProcessor::TextProcessor() : chunk_size_(500), chunk_overlap_(50), language_("chinese") {
        load_stop_words();
    }

    TextProcessor::~TextProcessor() {}

    void TextProcessor::set_chunk_parameters(size_t chunk_size, size_t chunk_overlap) {
        chunk_size_ = chunk_size;
        chunk_overlap_ = chunk_overlap;

        // 确保重叠大小不超过块大小
        if (chunk_overlap_ >= chunk_size_) {
            chunk_overlap_ = chunk_size_ / 2;
        }
    }

    std::string TextProcessor::clean_text(const std::string& text) {
        if (text.empty()) {
            return "";
        }

        std::string cleaned;
        cleaned.reserve(text.size());

        // 预处理：移除控制字符，保留基本可打印字符
        for (char c : text) {
            if (std::iscntrl(static_cast<unsigned char>(c)) &&
                !std::isspace(static_cast<unsigned char>(c))) {
                continue;
            }
            cleaned.push_back(c);
        }

        // 替换多个空白字符为单个空格
        std::regex whitespace_regex("\\s+");
        cleaned = std::regex_replace(cleaned, whitespace_regex, " ");

        // 移除多余的标点符号
        std::regex punctuation_regex("[,;:]+");
        cleaned = std::regex_replace(cleaned, punctuation_regex, ",");

        // 去除首尾空格
        cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r\f\v"));
        cleaned.erase(cleaned.find_last_not_of(" \t\n\r\f\v") + 1);

        return cleaned;
    }

    std::vector<TextChunk> TextProcessor::chunk_text(const std::string& text, int page_number) {
        std::vector<TextChunk> chunks;

        if (text.empty()) {
            return chunks;
        }

        // 清洗文本
        std::string cleaned_text = clean_text(text);

        // 如果文本长度小于等于块大小，直接作为一个块
        if (cleaned_text.length() <= chunk_size_) {
            chunks.emplace_back(cleaned_text, 0, cleaned_text.length(), page_number);
            return chunks;
        }

        // 首先尝试使用LangChain进行智能分块
        std::string langchain_cmd =
            "python python/langchain_processor.py --text \"" + escape_json_string(cleaned_text) +
            "\" --chunk-size " + std::to_string(chunk_size_) + " --chunk-overlap " +
            std::to_string(chunk_overlap_) + " --output langchain_chunks.json";

        FILE* pipe = popen(langchain_cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
            std::string result = "";
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
            pclose(pipe);

            // 尝试解析JSON结果
            if (!result.empty() && (result.find("{") == 0 || result.find("[") == 0)) {
                try {
                    // 简单解析分块结果
                    size_t chunks_start = result.find("\"chunks\":");
                    if (chunks_start != std::string::npos) {
                        size_t array_start = result.find("[", chunks_start);
                        size_t array_end = result.find("]", array_start);
                        if (array_start != std::string::npos && array_end != std::string::npos) {
                            std::string chunks_array =
                                result.substr(array_start + 1, array_end - array_start - 1);

                            // 解析块
                            size_t pos = 0;
                            while (pos < chunks_array.length()) {
                                size_t text_start = chunks_array.find("\"text\":", pos);
                                if (text_start == std::string::npos)
                                    break;

                                size_t value_start = chunks_array.find("\"", text_start + 7) + 1;
                                size_t value_end = chunks_array.find("\"", value_start);
                                if (value_end == std::string::npos)
                                    break;

                                std::string chunk_text =
                                    chunks_array.substr(value_start, value_end - value_start);
                                // 处理转义字符
                                size_t escape_pos = 0;
                                while ((escape_pos = chunk_text.find("\\n", escape_pos)) !=
                                       std::string::npos) {
                                    chunk_text.replace(escape_pos, 2, "\n");
                                    escape_pos += 1;
                                }

                                if (!chunk_text.empty()) {
                                    chunks.emplace_back(chunk_text, 0, chunk_text.length(),
                                                        page_number);
                                }

                                pos = value_end + 1;
                            }

                            if (!chunks.empty()) {
                                std::cout << "使用LangChain分块，共 " << chunks.size() << " 块"
                                          << std::endl;
                                return chunks;
                            }
                        }
                    }
                } catch (...) {
                    // 解析失败，继续使用原来的方法
                }
            }
        }

        // 如果LangChain分块失败，尝试使用简化版本
        std::cout << "LangChain分块失败，尝试简化版本..." << std::endl;
        std::string simple_cmd = "python python/simple_langchain_processor.py --text \"" +
                                 escape_json_string(cleaned_text) + "\" --chunk-size " +
                                 std::to_string(chunk_size_) + " --chunk-overlap " +
                                 std::to_string(chunk_overlap_) + " --output simple_chunks.json";

        FILE* simple_pipe = popen(simple_cmd.c_str(), "r");
        if (simple_pipe) {
            char buffer[128];
            std::string result = "";
            while (fgets(buffer, sizeof(buffer), simple_pipe) != nullptr) {
                result += buffer;
            }
            pclose(simple_pipe);

            // 尝试解析JSON结果
            if (!result.empty() && (result.find("{") == 0 || result.find("[") == 0)) {
                try {
                    // 简单解析分块结果
                    size_t chunks_start = result.find("\"chunks\":");
                    if (chunks_start != std::string::npos) {
                        size_t array_start = result.find("[", chunks_start);
                        size_t array_end = result.find("]", array_start);
                        if (array_start != std::string::npos && array_end != std::string::npos) {
                            std::string chunks_array =
                                result.substr(array_start + 1, array_end - array_start - 1);

                            // 解析块
                            size_t pos = 0;
                            while (pos < chunks_array.length()) {
                                size_t text_start = chunks_array.find("\"text\":", pos);
                                if (text_start == std::string::npos)
                                    break;

                                size_t value_start = chunks_array.find("\"", text_start + 7) + 1;
                                size_t value_end = chunks_array.find("\"", value_start);
                                if (value_end == std::string::npos)
                                    break;

                                std::string chunk_text =
                                    chunks_array.substr(value_start, value_end - value_start);
                                // 处理转义字符
                                size_t escape_pos = 0;
                                while ((escape_pos = chunk_text.find("\\n", escape_pos)) !=
                                       std::string::npos) {
                                    chunk_text.replace(escape_pos, 2, "\n");
                                    escape_pos += 1;
                                }

                                if (!chunk_text.empty()) {
                                    chunks.emplace_back(chunk_text, 0, chunk_text.length(),
                                                        page_number);
                                }

                                pos = value_end + 1;
                            }

                            if (!chunks.empty()) {
                                std::cout << "使用简化版分块，共 " << chunks.size() << " 块"
                                          << std::endl;
                                return chunks;
                            }
                        }
                    }
                } catch (...) {
                    // 解析失败，继续使用原来的方法
                }
            }
        }

        // 如果Python分块都失败，使用原来的C++分块方法
        std::cout << "Python分块失败，使用C++分块方法..." << std::endl;

        // 分割成句子
        std::vector<std::string> sentences = split_into_sentences(cleaned_text);

        size_t current_pos = 0;
        std::string current_chunk;

        for (const auto& sentence : sentences) {
            // 如果当前块加上新句子不超过块大小，添加到当前块
            if (current_chunk.length() + sentence.length() + 1 <= chunk_size_) {
                if (!current_chunk.empty()) {
                    current_chunk += " ";
                }
                current_chunk += sentence;
            } else {
                // 保存当前块（如果不为空）
                if (!current_chunk.empty()) {
                    size_t start_pos = current_pos - current_chunk.length();
                    chunks.emplace_back(current_chunk, start_pos, current_pos, page_number);
                }

                // 开始新块
                current_chunk = sentence;
                current_pos += sentence.length() + 1;  // +1 for space

                // 处理重叠
                if (!chunks.empty() && chunk_overlap_ > 0) {
                    // 获取前一个块的末尾部分
                    const TextChunk& prev_chunk = chunks.back();
                    std::string prev_text = prev_chunk.text;

                    // 找到合适的重叠点（尽量在句子边界）
                    size_t overlap_start = prev_text.length() > chunk_overlap_
                                               ? prev_text.length() - chunk_overlap_
                                               : 0;

                    // 找到下一个句子的开始
                    size_t sentence_start = prev_text.find(". ", overlap_start);
                    if (sentence_start != std::string::npos) {
                        overlap_start = sentence_start + 2;  // +2 for ". "
                    }

                    // 添加重叠部分到当前块
                    if (overlap_start < prev_text.length()) {
                        std::string overlap = prev_text.substr(overlap_start);
                        current_chunk = overlap + " " + current_chunk;
                    }
                }
            }

            current_pos += sentence.length() + 1;  // +1 for space
        }

        // 添加最后一个块（如果不为空）
        if (!current_chunk.empty()) {
            size_t start_pos = current_pos - current_chunk.length();
            chunks.emplace_back(current_chunk, start_pos, current_pos, page_number);
        }

        std::cout << "使用C++分块，共 " << chunks.size() << " 块" << std::endl;
        return chunks;
    }

    std::vector<TextChunk> TextProcessor::chunk_pages(const std::vector<std::string>& pages_text) {
        std::vector<TextChunk> all_chunks;

        for (size_t i = 0; i < pages_text.size(); ++i) {
            std::vector<TextChunk> page_chunks = chunk_text(pages_text[i], static_cast<int>(i));
            all_chunks.insert(all_chunks.end(), page_chunks.begin(), page_chunks.end());
        }

        return all_chunks;
    }

    std::vector<std::string> TextProcessor::extract_keywords(const std::string& text,
                                                             size_t max_keywords) {
        std::vector<std::string> keywords;

        if (text.empty()) {
            return keywords;
        }

        // 首先尝试使用LangChain进行高级关键词提取
        std::string langchain_cmd = "python python/langchain_processor.py --text \"" +
                                    escape_json_string(text) + "\" --max-keywords " +
                                    std::to_string(max_keywords) + " --keyword-method hybrid";

        FILE* pipe = popen(langchain_cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
            std::string result = "";
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
            pclose(pipe);

            // 尝试解析JSON结果
            if (!result.empty() && (result.find("{") == 0 || result.find("[") == 0)) {
                // 简单解析关键词数组
                size_t keywords_start = result.find("\"keywords\":");
                if (keywords_start != std::string::npos) {
                    size_t array_start = result.find("[", keywords_start);
                    size_t array_end = result.find("]", array_start);
                    if (array_start != std::string::npos && array_end != std::string::npos) {
                        std::string keywords_array =
                            result.substr(array_start + 1, array_end - array_start - 1);

                        // 解析关键词
                        size_t start = 0;
                        size_t end = keywords_array.find("\"", start);
                        while (end != std::string::npos && keywords.size() < max_keywords) {
                            start = end + 1;
                            end = keywords_array.find("\"", start);
                            if (end != std::string::npos) {
                                std::string keyword = keywords_array.substr(start, end - start);
                                if (!keyword.empty()) {
                                    keywords.push_back(keyword);
                                }
                                start = end + 1;
                                end = keywords_array.find("\"", start);
                            }
                        }
                    }
                }
            }
        }

        // 如果LangChain方法失败，尝试使用简化版本
        if (keywords.empty()) {
            std::cout << "LangChain processor failed, trying simple version..." << std::endl;
            std::string simple_cmd = "python python/simple_langchain_processor.py --text \"" +
                                     escape_json_string(text) + "\" --max-keywords " +
                                     std::to_string(max_keywords);

            FILE* simple_pipe = popen(simple_cmd.c_str(), "r");
            if (simple_pipe) {
                char buffer[128];
                std::string result = "";
                while (fgets(buffer, sizeof(buffer), simple_pipe) != nullptr) {
                    result += buffer;
                }
                pclose(simple_pipe);

                // 尝试解析JSON结果
                if (!result.empty() && (result.find("{") == 0 || result.find("[") == 0)) {
                    // 简单解析关键词数组
                    size_t keywords_start = result.find("\"keywords\":");
                    if (keywords_start != std::string::npos) {
                        size_t array_start = result.find("[", keywords_start);
                        size_t array_end = result.find("]", array_start);
                        if (array_start != std::string::npos && array_end != std::string::npos) {
                            std::string keywords_array =
                                result.substr(array_start + 1, array_end - array_start - 1);

                            // 解析关键词
                            size_t start = 0;
                            size_t end = keywords_array.find("\"", start);
                            while (end != std::string::npos && keywords.size() < max_keywords) {
                                start = end + 1;
                                end = keywords_array.find("\"", start);
                                if (end != std::string::npos) {
                                    std::string keyword = keywords_array.substr(start, end - start);
                                    if (!keyword.empty()) {
                                        keywords.push_back(keyword);
                                    }
                                    start = end + 1;
                                    end = keywords_array.find("\"", start);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 如果LangChain方法失败，回退到原来的方法
        if (keywords.empty()) {
            // 计算词频
            std::map<std::string, int> word_freq = calculate_word_frequency(text);

            // 过滤停用词和短词
            std::vector<std::pair<std::string, int>> filtered_words;
            for (const auto& [word, freq] : word_freq) {
                if (word.length() > 1 &&
                    std::find(stop_words_.begin(), stop_words_.end(), word) == stop_words_.end()) {
                    filtered_words.emplace_back(word, freq);
                }
            }

            // 按频率排序
            std::sort(filtered_words.begin(), filtered_words.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            // 取前max_keywords个词
            size_t count = std::min(max_keywords, filtered_words.size());
            for (size_t i = 0; i < count; ++i) {
                keywords.push_back(filtered_words[i].first);
            }
        }

        return keywords;
    }

    double TextProcessor::calculate_similarity(const std::string& text1, const std::string& text2) {
        if (text1.empty() || text2.empty()) {
            return 0.0;
        }

        // 简单的基于词汇重叠的相似度计算
        auto words1 = tokenize(text1);
        auto words2 = tokenize(text2);

        if (words1.empty() || words2.empty()) {
            return 0.0;
        }

        // 计算词集
        std::unordered_map<std::string, int> word_set1;
        for (const auto& word : words1) {
            word_set1[word]++;
        }

        std::unordered_map<std::string, int> word_set2;
        for (const auto& word : words2) {
            word_set2[word]++;
        }

        // 计算交集
        size_t intersection = 0;
        for (const auto& [word, count] : word_set1) {
            if (word_set2.find(word) != word_set2.end()) {
                intersection += std::min(count, word_set2[word]);
            }
        }

        // 计算并集
        size_t union_size = words1.size() + words2.size() - intersection;

        // 计算Jaccard相似度
        return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
    }

    void TextProcessor::set_language(const std::string& language) {
        language_ = language;
        load_stop_words();
    }

    const std::string& TextProcessor::get_language() const { return language_; }

    void TextProcessor::load_stop_words() {
        stop_words_.clear();

        if (language_ == "chinese") {
            // 中文停用词
            stop_words_ = {"的", "了", "在",   "是",   "我",   "有",   "和",   "就",   "不",
                           "人", "都", "一",   "一个", "上",   "也",   "很",   "到",   "说",
                           "要", "去", "你",   "会",   "着",   "没有", "看",   "好",   "自己",
                           "这", "那", "现在", "可以", "但是", "还是", "因为", "什么", "如果"};
        } else {
            // 英文停用词
            stop_words_ = {"the",     "a",      "an",      "and",     "or",      "but",  "if",
                           "because", "as",     "until",   "while",   "of",      "at",   "by",
                           "for",     "with",   "about",   "against", "between", "into", "through",
                           "during",  "before", "after",   "above",   "below",   "to",   "from",
                           "up",      "down",   "in",      "out",     "on",      "off",  "over",
                           "under",   "again",  "further", "then",    "once",    "here", "there",
                           "when",    "where",  "why",     "how",     "all",     "any",  "both",
                           "each",    "few",    "more",    "most",    "other",   "some", "such",
                           "no",      "nor",    "not",     "only",    "own",     "same", "so",
                           "than",    "too",    "very",    "s",       "t",       "can",  "will",
                           "just",    "don",    "should",  "now"};
        }
    }

    bool TextProcessor::is_punctuation(char c) const {
        return c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':' || c == '(' ||
               c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == '"' || c == '\'' ||
               c == '-' || c == '_';
    }

    bool TextProcessor::is_whitespace(char c) const {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    std::vector<std::string> TextProcessor::split_into_sentences(const std::string& text) {
        std::vector<std::string> sentences;

        if (text.empty()) {
            return sentences;
        }

        // 简单的句子分割（基于句号、问号、感叹号）
        std::regex sentence_regex("([.!?。！？])");
        std::sregex_token_iterator iter(text.begin(), text.end(), sentence_regex, -1);
        std::sregex_token_iterator end;

        std::string current_sentence;
        for (std::sregex_token_iterator i = iter; i != end; ++i) {
            std::string token = *i;
            current_sentence += token;

            // 如果token是句子结束符，保存当前句子
            if (token.length() == 1 && (token == "." || token == "!" || token == "?" ||
                                        token == "。" || token == "！" || token == "？")) {
                std::string trimmed = clean_text(current_sentence);
                if (!trimmed.empty()) {
                    sentences.push_back(trimmed);
                }
                current_sentence.clear();
            }
        }

        // 添加最后一个句子（如果不为空）
        if (!current_sentence.empty()) {
            std::string trimmed = clean_text(current_sentence);
            if (!trimmed.empty()) {
                sentences.push_back(trimmed);
            }
        }

        return sentences;
    }

    size_t TextProcessor::levenshtein_distance(const std::string& s1, const std::string& s2) {
        const size_t m = s1.size();
        const size_t n = s2.size();

        if (m == 0)
            return n;
        if (n == 0)
            return m;

        // 创建距离矩阵
        std::vector<std::vector<size_t>> dist(m + 1, std::vector<size_t>(n + 1));

        // 初始化第一行和第一列
        for (size_t i = 0; i <= m; ++i) dist[i][0] = i;
        for (size_t j = 0; j <= n; ++j) dist[0][j] = j;

        // 填充距离矩阵
        for (size_t i = 1; i <= m; ++i) {
            for (size_t j = 1; j <= n; ++j) {
                if (s1[i - 1] == s2[j - 1]) {
                    dist[i][j] = dist[i - 1][j - 1];
                } else {
                    dist[i][j] = 1 + std::min({dist[i - 1][j], dist[i][j - 1], dist[i - 1][j - 1]});
                }
            }
        }

        return dist[m][n];
    }

    std::map<std::string, int> TextProcessor::calculate_word_frequency(const std::string& text) {
        std::map<std::string, int> freq;

        if (text.empty()) {
            return freq;
        }

        // 分词
        auto words = tokenize(text);

        // 统计词频
        for (const auto& word : words) {
            freq[word]++;
        }

        return freq;
    }

    std::vector<std::string> TextProcessor::tokenize(const std::string& text) {
        std::vector<std::string> tokens;

        if (text.empty()) {
            return tokens;
        }

        std::string current_token;

        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                current_token += c;
            } else if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        }

        // 添加最后一个token（如果不为空）
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }

        return tokens;
    }

    std::string TextProcessor::escape_json_string(const std::string& input) {
        std::string result;
        result.reserve(input.size() * 2);  // 预留足够空间

        for (char c : input) {
            switch (c) {
            case '"':
                result += "\\\"";
                break;
            case '\\':
                result += "\\\\";
                break;
            case '\b':
                result += "\\b";
                break;
            case '\f':
                result += "\\f";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if (c >= 0 && c < 32) {
                    // 转义控制字符
                    char buffer[7];
                    snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    result += buffer;
                } else {
                    result += c;
                }
            }
        }

        return result;
    }

}  // namespace paper_qa