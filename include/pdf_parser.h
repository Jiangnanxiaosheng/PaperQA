#ifndef PDF_PARSER_H
#define PDF_PARSER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace paper_qa {

    /**
     * @brief PDF文档解析器类
     *
     * 负责解析PDF文件，提取文本内容和元数据
     */
    class PdfParser {
    public:
        /**
         * @brief 构造函数
         */
        PdfParser();

        /**
         * @brief 析构函数
         */
        ~PdfParser();

        /**
         * @brief 解析PDF文件
         *
         * @param file_path PDF文件路径
         * @return true 解析成功
         * @return false 解析失败
         */
        bool parse(const std::string& file_path);

        /**
         * @brief 获取提取的文本内容
         *
         * @return const std::string& 文本内容
         */
        const std::string& get_text() const;

        /**
         * @brief 获取PDF元数据
         *
         * @return const std::map<std::string, std::string>& 元数据键值对
         */
        const std::map<std::string, std::string>& get_metadata() const;

        /**
         * @brief 获取页数
         *
         * @return int 页数
         */
        int get_page_count() const;

        /**
         * @brief 获取指定页面的文本
         *
         * @param page_number 页码（从0开始）
         * @return std::string 页面文本
         */
        std::string get_page_text(int page_number) const;

        /**
         * @brief 获取所有页面的文本
         *
         * @return const std::vector<std::string>& 页面文本列表
         */
        const std::vector<std::string>& get_pages_text() const;

        /**
         * @brief 清空解析结果
         */
        void clear();

    private:
        // PDFium相关结构体
        struct PdfiumContext;

        // PDFium上下文
        std::unique_ptr<PdfiumContext> context_;

        // 提取的文本内容
        std::string text_;

        // 按页面存储的文本
        std::vector<std::string> pages_text_;

        // PDF元数据
        std::map<std::string, std::string> metadata_;

        // 页数
        int page_count_;

        /**
         * @brief 初始化PDFium库
         *
         * @return true 初始化成功
         * @return false 初始化失败
         */
        bool initialize_pdfium();

        /**
         * @brief 释放PDFium资源
         */
        void shutdown_pdfium();

        /**
         * @brief 从PDF文档中提取文本
         *
         * @return true 提取成功
         * @return false 提取失败
         */
        bool extract_text();

        /**
         * @brief 从PDF文档中提取元数据
         *
         * @return true 提取成功
         * @return false 提取失败
         */
        bool extract_metadata();

        /**
         * @brief 按页面提取文本
         *
         * @return true 提取成功
         * @return false 提取失败
         */
        bool extract_pages_text();
    };

}  // namespace paper_qa

#endif  // PDF_PARSER_H