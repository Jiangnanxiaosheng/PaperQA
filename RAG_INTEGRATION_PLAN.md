# RAG系统集成计划

## 当前问题分析

根据测试结果和用户反馈，当前系统存在以下问题：

1. **LangChain依赖问题**：导致复杂的兼容性和依赖问题
2. **多层回退机制复杂**：使系统流程难以理解和维护
3. **关键词提取效果不佳**：只有GLM4.5能提供高质量的关键词提取
4. **文本分块不理想**：15118字符的文本只分成1块，没有有效分块

## 简化方案设计

### 核心原则

1. **移除LangChain依赖**：简化系统，减少兼容性问题
2. **专注于有效组件**：主要使用GLM4.5和简化版处理器
3. **明确职责分工**：每个组件有明确的职责和功能
4. **简化执行流程**：使系统更易于理解和维护

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   C++ Core      │    │   Simple Text   │    │     GLM4.5      │
│                 │    │   Processor     │    │                 │
│ • PDF Parser    │◄──►│ • Text Chunking │◄──►│ • Keyword Ext.  │
│ • Main Program  │    │ • Basic Token.  │    │ • Semantic Anal.│
│ • RAG Engine    │    │ • Clean Text    │    │ • Q&A Support  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 详细实施计划

### 1. 简化文本处理器 (src/text_processor.cpp)

#### 修改目标
- 移除LangChain相关代码
- 直接使用简化版文本处理器
- 优化文本分块逻辑

#### 具体修改

**1.1 文本分块函数 (chunk_text)**
```cpp
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

    // 直接使用简化版文本处理器进行分块
    std::string simple_cmd = "python python/simple_langchain_processor.py --text \"" +
                            escape_json_string(cleaned_text) + "\" --chunk-size " +
                            std::to_string(chunk_size_) + " --chunk-overlap " +
                            std::to_string(chunk_overlap_) + " --output simple_chunks.json";

    FILE* pipe = popen(simple_cmd.c_str(), "r");
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
                        std::string chunks_array = result.substr(array_start + 1, array_end - array_start - 1);
                        
                        // 解析块
                        size_t pos = 0;
                        while (pos < chunks_array.length()) {
                            size_t text_start = chunks_array.find("\"text\":", pos);
                            if (text_start == std::string::npos) break;
                            
                            size_t value_start = chunks_array.find("\"", text_start + 7) + 1;
                            size_t value_end = chunks_array.find("\"", value_start);
                            if (value_end == std::string::npos) break;
                            
                            std::string chunk_text = chunks_array.substr(value_start, value_end - value_start);
                            // 处理转义字符
                            size_t escape_pos = 0;
                            while ((escape_pos = chunk_text.find("\\n", escape_pos)) != std::string::npos) {
                                chunk_text.replace(escape_pos, 2, "\n");
                                escape_pos += 1;
                            }
                            
                            if (!chunk_text.empty()) {
                                chunks.emplace_back(chunk_text, 0, chunk_text.length(), page_number);
                            }
                            
                            pos = value_end + 1;
                        }
                        
                        if (!chunks.empty()) {
                            std::cout << "使用简化版分块，共 " << chunks.size() << " 块" << std::endl;
                            return chunks;
                        }
                    }
                }
            } catch (...) {
                // 解析失败，继续使用C++分块方法
            }
        }
    }

    // 如果简化版分块失败，使用C++分块方法
    std::cout << "简化版分块失败，使用C++分块方法..." << std::endl;
    
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
```

**1.2 关键词提取函数 (extract_keywords)**
```cpp
std::vector<std::string> TextProcessor::extract_keywords(const std::string& text, size_t max_keywords) {
    std::vector<std::string> keywords;
    
    if (text.empty()) {
        return keywords;
    }
    
    // 直接使用GLM4.5进行关键词提取
    std::string glm_cmd = "python python/glm_interface.py --text \"" + 
                         escape_json_string(text) + "\" --operation keyword_extraction --max-keywords " + 
                         std::to_string(max_keywords) + " --output glm_keywords.json";
    
    FILE* pipe = popen(glm_cmd.c_str(), "r");
    if (pipe) {
        char buffer[128];
        std::string result = "";
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
        
        std::cout << "GLM4.5关键词提取完成" << std::endl;
        
        // 尝试解析JSON结果
        if (!result.empty() && (result.find("{") == 0 || result.find("[") == 0)) {
            try {
                // 简单解析关键词结果
                size_t keywords_start = result.find("\"keywords\":");
                if (keywords_start != std::string::npos) {
                    size_t array_start = result.find("[", keywords_start);
                    size_t array_end = result.find("]", array_start);
                    if (array_start != std::string::npos && array_end != std::string::npos) {
                        std::string keywords_array = result.substr(array_start + 1, array_end - array_start - 1);
                        
                        // 解析关键词
                        size_t start = 0;
                        size_t end = keywords_array.find("\"", start);
                        while (end != std::string::npos && keywords.size() < max_keywords) {
                            start = end + 1;
                            end = keywords_array.find("\"", start);
                            if (end != std::string::npos) {
                                std::string keyword = keywords_array.substr(start, end - start);
                                if (!keyword.empty()) {
                                    // 清理关键词，移除序号和点
                                    size_t dot_pos = keyword.find(". ");
                                    if (dot_pos != std::string::npos) {
                                        keyword = keyword.substr(dot_pos + 2);
                                    }
                                    keywords.push_back(keyword);
                                }
                                start = end + 1;
                                end = keywords_array.find("\"", start);
                            }
                        }
                        
                        if (!keywords.empty()) {
                            std::cout << "使用GLM4.5提取关键词，共 " << keywords.size() << " 个" << std::endl;
                            return keywords;
                        }
                    }
                }
            } catch (...) {
                // 解析失败，继续使用简化版
            }
        }
    }
    
    // 如果GLM4.5失败，使用简化版文本处理器
    std::cout << "GLM4.5关键词提取失败，使用简化版..." << std::endl;
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
            try {
                // 简单解析关键词数组
                size_t keywords_start = result.find("\"keywords\":");
                if (keywords_start != std::string::npos) {
                    size_t array_start = result.find("[", keywords_start);
                    size_t array_end = result.find("]", array_start);
                    if (array_start != std::string::npos && array_end != std::string::npos) {
                        std::string keywords_array = result.substr(array_start + 1, array_end - array_start - 1);
                        
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
                        
                        if (!keywords.empty()) {
                            std::cout << "使用简化版提取关键词，共 " << keywords.size() << " 个" << std::endl;
                            return keywords;
                        }
                    }
                }
            } catch (...) {
                // 解析失败，继续使用原来的方法
            }
        }
    }
    
    // 如果都失败，回退到原来的方法
    std::cout << "所有关键词提取方法失败，使用基本词频统计..." << std::endl;
    
    // 计算词频
    std::map<std::string, int> word_freq = calculate_word_frequency(text);
    
    // 过滤停用词和短词
    std::vector<std::pair<std::string, int>> filtered_words;
    for (const auto& [word, freq] : word_freq) {
        if (word.length() > 2 && 
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
    
    std::cout << "使用基本词频统计提取关键词，共 " << keywords.size() << " 个" << std::endl;
    return keywords;
}
```

### 2. 优化简化版文本处理器 (python/simple_langchain_processor.py)

#### 修改目标
- 改进文本分块逻辑，确保长文本被合理分块
- 优化关键词提取，过滤更多无意义词汇

#### 具体修改

**2.1 改进分块逻辑**
```python
def simple_chunking(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    改进的简单文本分块
    
    Args:
        text: 输入文本
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        分块结果列表
    """
    result = []
    start = 0
    
    # 对于长文本，确保至少分成多个块
    min_chunks = max(1, len(text) // (chunk_size * 2))
    
    while start < len(text):
        # 计算块结束位置
        end = start + chunk_size
        
        # 如果不是最后一块，尝试在句子边界处分割
        if end < len(text):
            # 寻找最近的句子结束符
            for i in range(end, max(start, end - 100), -1):
                if text[i] in '.!?\n。！？':
                    end = i + 1
                    break
        
        # 确保不超过文本长度
        end = min(end, len(text))
        
        # 提取块
        chunk = text[start:end].strip()
        if chunk:
            result.append({
                "text": chunk,
                "chunk_id": len(result),
                "metadata": {"method": "simple"}
            })
        
        # 计算下一个块的开始位置（考虑重叠）
        start = end - chunk_overlap if end - chunk_overlap > start else end
        
        # 如果已经达到最小块数但文本还有剩余，调整块大小
        if len(result) >= min_chunks and start < len(text):
            # 增加块大小以减少块数
            chunk_size = int(chunk_size * 1.5)
    
    print(f"Simple chunking completed with {len(result)} chunks")
    return result
```

### 3. RAG系统集成

#### 3.1 添加向量化支持
```cpp
class VectorDatabase {
private:
    std::vector<std::vector<float>> embeddings;
    std::vector<std::string> texts;
    std::vector<TextChunk> chunks;
    
public:
    void add_embedding(const std::vector<float>& embedding, const std::string& text, const TextChunk& chunk) {
        embeddings.push_back(embedding);
        texts.push_back(text);
        chunks.push_back(chunk);
    }
    
    std::vector<std::pair<int, float>> search(const std::vector<float>& query_embedding, int top_k = 5) {
        std::vector<std::pair<int, float>> results;
        
        for (size_t i = 0; i < embeddings.size(); ++i) {
            // 计算余弦相似度
            float similarity = cosine_similarity(query_embedding, embeddings[i]);
            results.emplace_back(i, similarity);
        }
        
        // 按相似度排序
        std::sort(results.begin(), results.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 返回前top_k个结果
        if (results.size() > top_k) {
            results.resize(top_k);
        }
        
        return results;
    }
    
private:
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return 0.0f;
        
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        for (size_t i = 0; i < a.size(); ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
        
        return dot_product / (sqrt(norm_a) * sqrt(norm_b));
    }
};
```

#### 3.2 修改主程序 (src/main.cpp)
```cpp
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pdf_file>" << std::endl;
        return 1;
    }

    std::string pdf_file = argv[1];

    try {
        // 解析PDF
        PDFParser pdf_parser;
        auto pdf_result = pdf_parser.parse_pdf(pdf_file);

        std::cout << "PDF解析成功!" << std::endl;
        std::cout << "页数: " << pdf_result.page_count << std::endl;
        std::cout << std::endl;

        std::cout << "元数据:" << std::endl;
        std::cout << "  file_path: " << pdf_result.metadata.at("file_path") << std::endl;
        std::cout << "  page_count: " << pdf_result.metadata.at("page_count") << std::endl;
        std::cout << std::endl;

        // 处理文本
        TextProcessor text_processor;
        text_processor.set_chunk_parameters(1000, 100);  // 增加块大小和重叠

        std::cout << "使用GLM4.5增强的文本处理..." << std::endl;
        std::cout << std::endl;

        // 分块处理
        auto chunks = text_processor.chunk_pages(pdf_result.pages_text);

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
        for (const auto& page_text : pdf_result.pages_text) {
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
        std::string query = "impulse-like periodic correlations";
        std::cout << "示例查询: \"" << query << "\"" << std::endl;
        
        // 创建查询嵌入（示例中使用随机向量）
        std::vector<float> query_embedding(384, 0.0f);
        for (auto& val : query_embedding) {
            val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        
        // 搜索相关文本
        auto search_results = vector_db.search(query_embedding, 3);
        
        std::cout << "检索结果:" << std::endl;
        for (const auto& [idx, similarity] : search_results) {
            std::cout << "  相关度: " << similarity << std::endl;
            std::cout << "  内容: " << chunks[idx].text.substr(0, 100) << "..." << std::endl;
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

## 预期效果

1. **简化的系统架构**：移除LangChain依赖，减少兼容性问题
2. **更好的文本分块**：确保长文本被合理分成多个块
3. **高质量的关键词提取**：主要使用GLM4.5，提供更准确的关键词
4. **RAG功能集成**：添加向量化和检索功能，支持问答和语义搜索

## 实施步骤

1. 修改 `src/text_processor.cpp` 中的 `chunk_text` 和 `extract_keywords` 函数
2. 优化 `python/simple_langchain_processor.py` 中的分块逻辑
3. 添加向量化支持和RAG功能
4. 更新 `src/main.cpp` 以集成RAG系统
5. 测试验证系统功能

这个简化方案将使系统更加稳定、高效，同时保持强大的文本处理和检索能力。