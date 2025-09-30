#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace paper_qa {

    /**
     * @brief 文本块结构体
     */
    struct TextChunk {
        std::string text;                             // 文本内容
        size_t start_pos;                             // 在原文中的起始位置
        size_t end_pos;                               // 在原文中的结束位置
        int page_number;                              // 页码（从0开始）
        std::map<std::string, std::string> metadata;  // 附加元数据

        TextChunk() : start_pos(0), end_pos(0), page_number(-1) {}
        TextChunk(const std::string& t, size_t start, size_t end, int page = -1)
            : text(t), start_pos(start), end_pos(end), page_number(page) {}
    };

    /**
     * @brief 文本处理器类
     *
     * 负责文本清洗、分块和预处理
     */
    class TextProcessor {
    public:
        /**
         * @brief 构造函数
         */
        TextProcessor();

        /**
         * @brief 析构函数
         */
        ~TextProcessor();

        /**
         * @brief 设置分块参数
         *
         * @param chunk_size 块大小（字符数）
         * @param chunk_overlap 块重叠大小（字符数）
         */
        void set_chunk_parameters(size_t chunk_size, size_t chunk_overlap);

        /**
         * @brief 清洗文本
         *
         * @param text 原始文本
         * @return std::string 清洗后的文本
         */
        std::string clean_text(const std::string& text);

        /**
         * @brief 将文本分块
         *
         * @param text 输入文本
         * @param page_number 页码（可选）
         * @return std::vector<TextChunk> 文本块列表
         */
        std::vector<TextChunk> chunk_text(const std::string& text, int page_number = -1);

        /**
         * @brief 将多页文本分块
         *
         * @param pages_text 多页文本列表
         * @return std::vector<TextChunk> 文本块列表
         */
        std::vector<TextChunk> chunk_pages(const std::vector<std::string>& pages_text);

        /**
         * @brief 提取文本关键词
         *
         * @param text 输入文本
         * @param max_keywords 最大关键词数量
         * @return std::vector<std::string> 关键词列表
         */
        std::vector<std::string> extract_keywords(const std::string& text,
                                                  size_t max_keywords = 10);

        /**
         * @brief 计算文本相似度
         *
         * @param text1 文本1
         * @param text2 文本2
         * @return double 相似度分数（0-1）
         */
        double calculate_similarity(const std::string& text1, const std::string& text2);

        /**
         * @brief 设置语言
         *
         * @param language 语言代码（如"chinese", "english"）
         */
        void set_language(const std::string& language);

        /**
         * @brief 获取当前语言设置
         *
         * @return const std::string& 语言代码
         */
        const std::string& get_language() const;

    private:
        // 分块参数
        size_t chunk_size_;
        size_t chunk_overlap_;

        // 语言设置
        std::string language_;

        // 停用词列表
        std::vector<std::string> stop_words_;

        /**
         * @brief 加载停用词
         */
        void load_stop_words();

        /**
         * @brief 检查字符是否是标点符号
         *
         * @param c 字符
         * @return true 是标点符号
         * @return false 不是标点符号
         */
        bool is_punctuation(char c) const;

        /**
         * @brief 检查字符是否是空白字符
         *
         * @param c 字符
         * @return true 是空白字符
         * @return false 不是空白字符
         */
        bool is_whitespace(char c) const;

        /**
         * @brief 在句子边界处分割文本
         *
         * @param text 输入文本
         * @return std::vector<std::string> 句子列表
         */
        std::vector<std::string> split_into_sentences(const std::string& text);

        /**
         * @brief 计算两个字符串的编辑距离
         *
         * @param s1 字符串1
         * @param s2 字符串2
         * @return size_t 编辑距离
         */
        size_t levenshtein_distance(const std::string& s1, const std::string& s2);

        /**
         * @brief 计算词频
         *
         * @param text 输入文本
         * @return std::map<std::string, int> 词频字典
         */
        std::map<std::string, int> calculate_word_frequency(const std::string& text);

        /**
         * @brief 分词
         *
         * @param text 输入文本
         * @return std::vector<std::string> 词语列表
         */
        std::vector<std::string> tokenize(const std::string& text);

    public:
        /**
         * @brief 转义JSON字符串中的特殊字符
         *
         * @param input 输入字符串
         * @return std::string 转义后的字符串
         */
        std::string escape_json_string(const std::string& input);
    };

}  // namespace paper_qa

#endif  // TEXT_PROCESSOR_H