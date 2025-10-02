#include "pdf_parser.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>  // For std::unique_ptr
#include <sstream>

#include "json.hpp"

// 为了方便使用，使用 nlohmann 命名空间
using json = nlohmann::json;

namespace paper_qa {

    // PDFium上下文结构体定义 (虽然这里没用到，但保留结构)
    struct PdfParser::PdfiumContext {
        std::string filepath;

        PdfiumContext() : filepath("") {}
    };

    PdfParser::PdfParser() : context_(std::make_unique<PdfiumContext>()), page_count_(0) {}

    PdfParser::~PdfParser() { clear(); }

    bool PdfParser::parse(const std::string& filepath) {
        // 清空之前的解析结果
        clear();

        // 检查文件是否存在
        std::ifstream file(filepath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "PDF file not found: " << filepath << std::endl;
            return false;
        }
        file.close();

        // 构建Python脚本的命令
        // 确保python3在你的PATH中，并且脚本路径正确
        std::string command = "python3 python/pdf_parser.py \"" + filepath + "\"";

        // 执行命令并打开管道读取输出
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to execute Python script." << std::endl;
            return false;
        }

        // 读取Python脚本的全部输出 (JSON字符串)
        std::stringstream ss;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            ss << buffer;
        }

        // 关闭管道并检查退出状态
        int exit_code = pclose(pipe);
        if (exit_code != 0) {
            std::cerr << "Python script exited with error code: " << exit_code << std::endl;
            // 尝试打印错误输出
            std::cerr << "Script output was: " << ss.str() << std::endl;
            return false;
        }

        std::string result = ss.str();
        if (result.empty()) {
            std::cerr << "Python script produced no output." << std::endl;
            return false;
        }

        // --- 使用 nlohmann/json 进行解析 ---
        try {
            // 1. 将字符串解析为JSON对象
            json data = json::parse(result);

            // 2. 安全地提取 "text" 字段
            if (data.contains("text") && data["text"].is_string()) {
                text_ = data["text"].get<std::string>();
            } else {
                std::cerr << "Warning: JSON does not contain a valid 'text' field." << std::endl;
            }

            // 3. 安全地提取 "page_count" 字段 (修正字段名)
            if (data.contains("page_count") && data["page_count"].is_number_integer()) {
                page_count_ = data["page_count"].get<int>();
            } else {
                std::cerr << "Warning: JSON does not contain a valid 'page_count' field."
                          << std::endl;
            }

            // 4. 安全地提取 "metadata" 对象
            if (data.contains("metadata") && data["metadata"].is_object()) {
                for (auto& [key, value] : data["metadata"].items()) {
                    // 确保值是字符串类型
                    if (value.is_string()) {
                        metadata_[key] = value.get<std::string>();
                    }
                }
            } else {
                std::cerr << "Warning: JSON does not contain a valid 'metadata' object."
                          << std::endl;
            }

            // 5. 安全地提取 "pages" 数组 (修正字段名)
            if (data.contains("pages") && data["pages"].is_array()) {
                pages_text_.clear();
                for (const auto& page_text_json : data["pages"]) {
                    if (page_text_json.is_string()) {
                        pages_text_.push_back(page_text_json.get<std::string>());
                    }
                }
            } else {
                std::cerr << "Warning: JSON does not contain a valid 'pages' array." << std::endl;
            }

        } catch (const json::parse_error& e) {
            // 捕获JSON解析错误
            std::cerr << "JSON Parse Error: " << e.what() << std::endl;
            std::cerr << "Failed to parse output: " << result << std::endl;
            return false;
        } catch (const std::exception& e) {
            // 捕获其他可能的异常
            std::cerr << "Error during JSON processing: " << e.what() << std::endl;
            return false;
        }

        // 如果没有致命错误，解析成功
        return true;
    }

    // --- Getter 函数实现 ---

    const std::string& PdfParser::get_text() const { return text_; }

    const std::map<std::string, std::string>& PdfParser::get_metadata() const { return metadata_; }

    int PdfParser::get_page_count() const { return page_count_; }

    std::string PdfParser::get_page_text(int pagenumber) const {
        // pagenumber 是从0开始的索引
        if (pagenumber < 0 || pagenumber >= static_cast<int>(pages_text_.size())) {
            return "";
        }
        return pages_text_[pagenumber];
    }

    const std::vector<std::string>& PdfParser::get_pages_text() const { return pages_text_; }

    void PdfParser::clear() {
        text_.clear();
        pages_text_.clear();
        metadata_.clear();
        page_count_ = 0;
        context_->filepath = "";
    }

}  // namespace paperqa
