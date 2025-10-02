#include "pdf_parser.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace paper_qa {

    // PDFium上下文结构体定义
    struct PdfParser::PdfiumContext {
        std::string file_path;

        PdfiumContext() : file_path("") {}
    };

    PdfParser::PdfParser() : context_(std::make_unique<PdfiumContext>()), page_count_(0) {}

    PdfParser::~PdfParser() { clear(); }

    bool PdfParser::parse(const std::string& file_path) {
        // 清空之前的解析结果
        clear();

        // 检查文件是否存在
        std::ifstream file(file_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "PDF file not found: " << file_path << std::endl;
            return false;
        }

        // 保存文件路径
        context_->file_path = file_path;

        // 使用Python脚本解析PDF
        std::string python_cmd = "python python/pdf_parser.py \"" + file_path + "\"";
        FILE* pipe = popen(python_cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to run Python script" << std::endl;
            return false;
        }

        // 读取Python脚本的输出
        char buffer[128];
        std::string result = "";
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }

        // 关闭管道
        pclose(pipe);

        // 解析结果
        if (result.empty()) {
            std::cerr << "Python script returned empty result" << std::endl;
            return false;
        }

        // 尝试解析JSON格式的结果
        // 首先检查是否是JSON格式
        if (result.find("{") == 0 || result.find("[") == 0) {
            // 简单的JSON解析（在实际应用中应该使用JSON解析库）
            size_t text_start = result.find("\"text\":");
            if (text_start != std::string::npos) {
                size_t text_value_start = result.find("\"", text_start + 7) + 1;
                size_t text_value_end = result.find("\"", text_value_start);
                if (text_value_end != std::string::npos) {
                    text_ = result.substr(text_value_start, text_value_end - text_value_start);
                    // 处理转义字符
                    size_t pos = 0;
                    while ((pos = text_.find("\\n", pos)) != std::string::npos) {
                        text_.replace(pos, 2, "\n");
                        pos += 1;
                    }
                }
            }

            // 解析页面文本
            size_t pages_start = result.find("\"pages\":");
            if (pages_start != std::string::npos) {
                size_t array_start = result.find("[", pages_start);
                size_t array_end = result.find("]", array_start);
                if (array_start != std::string::npos && array_end != std::string::npos) {
                    std::string pages_array =
                        result.substr(array_start + 1, array_end - array_start - 1);

                    // 简单解析页面数组
                    size_t start = 0;
                    size_t end = pages_array.find("\"", start);
                    while (end != std::string::npos) {
                        start = end + 1;
                        end = pages_array.find("\"", start);
                        if (end != std::string::npos) {
                            std::string page_text = pages_array.substr(start, end - start);
                            // 处理转义字符
                            size_t pos = 0;
                            while ((pos = page_text.find("\\n", pos)) != std::string::npos) {
                                page_text.replace(pos, 2, "\n");
                                pos += 1;
                            }
                            pages_text_.push_back(page_text);

                            // 跳过分隔符
                            start = end + 1;
                            end = pages_array.find("\"", start);
                        }
                    }
                }
            }

            // 如果没有解析到页面文本，使用整个文本作为单页
            if (pages_text_.empty() && !text_.empty()) {
                pages_text_.push_back(text_);
            }
        } else {
            // 如果不是JSON格式，使用原来的处理方式
            text_ = result;

            // 简单的文本分页（按双换行符分割，表示段落分隔）
            std::stringstream ss(text_);
            std::string line;
            std::string current_page;

            while (std::getline(ss, line)) {
                if (line.empty() && !current_page.empty()) {
                    pages_text_.push_back(current_page);
                    current_page = "";
                } else if (!line.empty()) {
                    if (!current_page.empty()) {
                        current_page += "\n";
                    }
                    current_page += line;
                }
            }

            // 添加最后一页
            if (!current_page.empty()) {
                pages_text_.push_back(current_page);
            }
        }

        page_count_ = pages_text_.size();

        // 添加一些基本的元数据
        metadata_["file_path"] = file_path;
        metadata_["page_count"] = std::to_string(page_count_);

        return true;
    }

    const std::string& PdfParser::get_text() const { return text_; }

    const std::map<std::string, std::string>& PdfParser::get_metadata() const { return metadata_; }

    int PdfParser::get_page_count() const { return page_count_; }

    std::string PdfParser::get_page_text(int page_number) const {
        if (page_number < 0 || page_number >= page_count_) {
            return "";
        }
        return pages_text_[page_number];
    }

    const std::vector<std::string>& PdfParser::get_pages_text() const { return pages_text_; }

    void PdfParser::clear() {
        text_.clear();
        pages_text_.clear();
        metadata_.clear();
        page_count_ = 0;
        context_->file_path = "";
    }

}  // namespace paper_qa