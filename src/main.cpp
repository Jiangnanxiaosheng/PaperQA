#include <iostream>
#include <string>

// #include "llm_client.h"
#include "pdf_parser.h"
// #include "rag_engine.h"
#include "text_processor.h"
// #include "vector_db.h"
// #include "vectorizer.h"
// #include "web_server.h"
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <pdf_file_path>" << std::endl;
        return 1;
    }

    std::string pdf_path = argv[1];

    // 创建PDF解析器
    paper_qa::PdfParser parser;

    // 解析PDF文件
    if (!parser.parse(pdf_path)) {
        std::cerr << "Failed to parse PDF file: " << pdf_path << std::endl;
        return 1;
    }

    // 获取并显示基本信息
    std::cout << "PDF解析成功!" << std::endl;
    std::cout << "页数: " << parser.get_page_count() << std::endl;

    // 显示元数据
    const auto& metadata = parser.get_metadata();
    std::cout << "\n元数据:" << std::endl;
    for (const auto& [key, value] : metadata) {
        std::cout << "  " << key << ": " << value << std::endl;
    }

    // 创建文本处理器
    paper_qa::TextProcessor text_processor;
    text_processor.set_chunk_parameters(500, 50);  // 块大小500，重叠50

    // 设置语言（根据PDF内容自动判断或手动设置）
    text_processor.set_language("english");  // 根据实际PDF内容设置

    std::cout << "\n使用LangChain和GLM4.5增强的文本处理..." << std::endl;

    // 获取所有页面文本
    const auto& pages_text = parser.get_pages_text();

    // 文本分块
    auto chunks = text_processor.chunk_pages(pages_text);

    std::cout << "\n文本分块结果:" << std::endl;
    std::cout << "总块数: " << chunks.size() << std::endl;

    // 显示前几个块的详细信息
    int preview_chunks = std::min(5, (int)chunks.size());
    for (int i = 0; i < preview_chunks; ++i) {
        const auto& chunk = chunks[i];
        std::cout << "\n块 " << (i + 1) << ":" << std::endl;
        std::cout << "  页码: " << (chunk.page_number + 1) << std::endl;
        std::cout << "  长度: " << chunk.text.length() << " 字符" << std::endl;
        std::cout << "  内容预览: " << chunk.text.substr(0, std::min(100, (int)chunk.text.length()))
                  << "..." << std::endl;
    }

    // 如果有更多块，显示提示
    if (chunks.size() > preview_chunks) {
        std::cout << "\n... 还有 " << (chunks.size() - preview_chunks) << " 块" << std::endl;
    }

    // 提取关键词
    if (!pages_text.empty()) {
        std::string full_text = parser.get_text();
        auto keywords = text_processor.extract_keywords(full_text, 10);

        std::cout << "\n关键词提取结果 (使用LangChain增强):" << std::endl;
        for (size_t i = 0; i < keywords.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << keywords[i] << std::endl;
        }

        // 如果有GLM4.5 API密钥，尝试使用GLM4.5进行高级关键词提取
        const char* glm_api_key = std::getenv("GLM_API_KEY");
        if (glm_api_key && keywords.size() > 0) {
            std::cout << "\n尝试使用GLM4.5进行高级关键词提取..." << std::endl;
            std::string glm_cmd =
                "python python/glm_interface.py --text \"" +
                text_processor.escape_json_string(full_text.substr(0, 1000)) +
                "\" --operation keyword_extraction --max-keywords 10 --output glm_keywords.json";

            int result = std::system(glm_cmd.c_str());
            if (result == 0) {
                std::cout << "GLM4.5关键词提取完成，结果保存在 glm_keywords.json" << std::endl;
            } else {
                std::cout << "GLM4.5关键词提取失败，使用LangChain结果" << std::endl;
            }
        }
    }

    // 测试文本相似度
    if (chunks.size() >= 2) {
        double similarity = text_processor.calculate_similarity(chunks[0].text, chunks[1].text);
        std::cout << "\n文本相似度测试:" << std::endl;
        std::cout << "块1和块2的相似度: " << similarity << std::endl;
    }
}