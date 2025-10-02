#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "llm_client.h"
#include "pdf_parser.h"
// #include "rag_engine.h"
#include "text_processor.h"
#include "vector_db.h"
// #include "vectorizer.h"
// #include "web_server.h"

using namespace paper_qa;

// 函数：转义JSON字符串
std::string escape_json_string(const std::string& input) {
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

// 函数：检查文件扩展名
std::string get_file_extension(const std::string& filename) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        return filename.substr(dot_pos + 1);
    }
    return "";
}

// 函数：读取TXT文件
std::string read_txt_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file>" << std::endl;
        std::cerr << "支持PDF和TXT文件" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];
    std::string extension = get_file_extension(file_path);

    // 转换为小写以便比较
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    try {
        std::vector<std::string> pages_text;
        std::string file_type;

        if (extension == "pdf") {
            // 解析PDF
            PdfParser pdf_parser;
            if (!pdf_parser.parse(file_path)) {
                std::cerr << "PDF解析失败!" << std::endl;
                return 1;
            }

            std::cout << "PDF解析成功!" << std::endl;
            std::cout << "页数: " << pdf_parser.get_page_count() << std::endl;
            std::cout << std::endl;

            std::cout << "元数据:" << std::endl;
            const auto& metadata = pdf_parser.get_metadata();
            for (const auto& [key, value] : metadata) {
                std::cout << "  " << key << ": " << value << std::endl;
            }
            std::cout << std::endl;

            pages_text = pdf_parser.get_pages_text();
            file_type = "PDF";
        } else if (extension == "txt") {
            // 读取TXT文件
            std::string content = read_txt_file(file_path);

            std::cout << "TXT文件读取成功!" << std::endl;
            std::cout << "文件大小: " << content.length() << " 字符" << std::endl;
            std::cout << std::endl;

            // 将整个TXT文件作为一个页面
            pages_text.push_back(content);
            file_type = "TXT";
        } else {
            std::cerr << "不支持的文件类型: " << extension << std::endl;
            std::cerr << "只支持PDF和TXT文件" << std::endl;
            return 1;
        }

        // 处理文本
        TextProcessor text_processor;
        text_processor.set_chunk_parameters(1000, 100);  // 增加块大小和重叠

        std::cout << "使用GLM4.5增强的文本处理..." << std::endl;
        std::cout << std::endl;

        // 分块处理
        auto chunks = text_processor.chunk_pages(pages_text);

        std::cout << "文本分块结果:" << std::endl;
        std::cout << "总块数: " << chunks.size() << std::endl;
        std::cout << std::endl;

        for (size_t i = 0; i < std::min(chunks.size(), static_cast<size_t>(3)); ++i) {
            std::cout << "块 " << (i + 1) << ":" << std::endl;
            std::cout << "  页码: " << chunks[i].page_number << std::endl;
            std::cout << "  长度: " << chunks[i].text.length() << " 字符" << std::endl;
            std::cout << "  内容预览: " << chunks[i].text.substr(0, 100) << "..." << std::endl;
            std::cout << std::endl;
        }

        // 提取关键词
        std::string full_text;
        for (const auto& page_text : pages_text) {
            full_text += page_text + "\n";
        }

        auto keywords = text_processor.extract_keywords(full_text, 10);

        std::cout << "关键词提取结果 (使用GLM4.5增强):" << std::endl;
        for (size_t i = 0; i < keywords.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << keywords[i] << std::endl;
        }
        std::cout << std::endl;

        // RAG处理
        std::cout << "初始化RAG系统..." << std::endl;
        VectorDatabase vector_db;

        // 为每个块创建嵌入并存储
        for (const auto& chunk : chunks) {
            // 这里应该调用嵌入模型API获取向量
            // 为了示例，我们使用随机向量
            std::vector<float> embedding(384, 0.0f);  // 假设384维嵌入
            for (auto& val : embedding) {
                val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }

            vector_db.add_embedding(embedding, chunk.text, chunk);
        }

        std::cout << "RAG系统初始化完成，已存储 " << chunks.size() << " 个文本块" << std::endl;

        // 示例查询
        std::string query = "AI伴侣叫什么，它的性格特点是什么？";
        std::cout << "示例查询: \"" << query << "\"" << std::endl;

        // 创建查询嵌入（示例中使用随机向量）
        std::vector<float> query_embedding(384, 0.0f);
        for (auto& val : query_embedding) {
            val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        // 搜索相关文本
        auto search_results = vector_db.search(query_embedding, 3);

        std::cout << "检索结果:" << std::endl;
        std::string context;
        for (const auto& [idx, similarity] : search_results) {
            std::cout << "  相关度: " << similarity << std::endl;
            std::cout << "  内容: " << chunks[idx].text.substr(0, 100) << "..." << std::endl;
            context += chunks[idx].text + "\n\n";
            std::cout << std::endl;
        }

        // 使用GLM4.5回答问题
        std::cout << "使用GLM4.5生成回答..." << std::endl;
        std::string glm_cmd = "python python/glm_interface.py --text \"" +
                              escape_json_string(context) +
                              "\" --operation answer_question --question \"" +
                              escape_json_string(query) + "\" --output glm_answer.json";

        FILE* pipe = popen(glm_cmd.c_str(), "r");
        if (pipe) {
            char buffer[128];
            std::string result = "";
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
            pclose(pipe);

            std::cout << "GLM4.5回答生成完成" << std::endl;

            // 尝试读取输出文件
            std::ifstream answer_file("glm_answer.json");
            if (answer_file.is_open()) {
                std::string json_content;
                std::string line;
                while (std::getline(answer_file, line)) {
                    json_content += line;
                }
                answer_file.close();

                // 尝试解析JSON结果
                if (!json_content.empty()) {
                    try {
                        // 简单解析回答结果
                        size_t answer_start = json_content.find("\"answer\":");
                        if (answer_start != std::string::npos) {
                            size_t value_start = json_content.find("\"", answer_start + 9) + 1;
                            size_t value_end = json_content.find("\"", value_start);
                            if (value_end != std::string::npos) {
                                std::string answer =
                                    json_content.substr(value_start, value_end - value_start);
                                // 处理转义字符
                                size_t escape_pos = 0;
                                while ((escape_pos = answer.find("\\n", escape_pos)) !=
                                       std::string::npos) {
                                    answer.replace(escape_pos, 2, "\n");
                                    escape_pos += 1;
                                }

                                std::cout << "回答:" << std::endl;
                                std::cout << "  " << answer << std::endl;
                            }
                        }
                    } catch (...) {
                        std::cout << "无法解析GLM4.5的回答" << std::endl;
                    }
                }
            } else {
                std::cout << "无法读取GLM4.5的回答文件" << std::endl;
            }
        }

        std::cout << file_type << "文件处理完成!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}