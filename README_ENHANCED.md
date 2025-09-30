# 增强版RAG系统使用说明

## 概述

本项目已经通过集成LangChain和GLM4.5模型进行了显著增强，解决了原始系统中关键词提取不准确和分词分段不精确的问题。

## 主要改进

### 1. 智能关键词提取
- **原始问题**: 提取的关键词无意义（如"the", "and", "of"等常见词）
- **解决方案**: 
  - 使用LangChain的TF-IDF和嵌入向量混合方法
  - 支持GLM4.5模型进行语义理解的关键词提取
  - 自动过滤停用词和无意义词汇

### 2. 语义文本分块
- **原始问题**: 简单基于字符长度的分块，破坏语义完整性
- **解决方案**:
  - 使用LangChain的语义分块器(SemanticChunker)
  - 基于句子边界的智能分割
  - 支持可配置的重叠大小，保持上下文连续性

### 3. GLM4.5模型集成
- **功能**: 提供高级文本理解和处理能力
- **应用场景**:
  - 学术论文关键词提取
  - 技术文档结构分析
  - 智能问答系统

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   C++ Core      │    │   LangChain     │    │     GLM4.5      │
│                 │    │                 │    │                 │
│ • PDF Parser    │◄──►│ • Text Splitter │◄──►│ • Keyword Ext.  │
│ • Text Processor│    │ • Embeddings    │    │ • Summarization │
│ • Main Program  │    │ • TF-IDF        │    │ • Q&A           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 安装和设置

### 1. 环境要求
- C++17兼容的编译器
- Python 3.8+
- CMake 3.16+

### 2. 安装Python依赖
```bash
# 安装基础依赖
pip install -r python/requirements.txt

# 可选：下载spaCy语言模型
python -m spacy download en_core_web_sm  # 英文
python -m spacy download zh_core_web_sm  # 中文
```

### 3. 设置GLM4.5 API（可选）
```bash
# 设置API密钥
export GLM_API_KEY="your_glm_api_key_here"

# 可选：设置自定义API端点
export GLM_BASE_URL="https://your-custom-endpoint.com/api/paas/v4"
```

### 4. 构建项目
```bash
# 创建构建目录
mkdir -p build && cd build

# 配置项目
cmake ..

# 编译
make

# 或者使用便捷目标（会自动安装Python依赖）
make setup_python_env
```

## 使用方法

### 基本使用（处理PDF文件）
```bash
./build/paper-qa path/to/your/document.pdf
```

### 高级使用（使用GLM4.5增强功能）
```bash
# 设置API密钥后运行
export GLM_API_KEY="your_api_key"
./build/paper-qa path/to/your/document.pdf
```

### 单独测试组件

#### 1. 测试LangChain文本处理器
```bash
# 处理文本文件
python python/langchain_processor.py --file input.txt --output output.json

# 直接处理文本
python python/langchain_processor.py --text "Your text here" --max-keywords 10
```

#### 2. 测试GLM4.5接口
```bash
# 关键词提取
python python/glm_interface.py --text "Your text here" --operation keyword_extraction

# 语义分块
python python/glm_interface.py --text "Your text here" --operation semantic_chunking

# 文本摘要
python python/glm_interface.py --text "Your text here" --operation summarization
```

#### 3. 运行完整测试套件
```bash
python test_enhanced_system.py
```

## 配置选项

### LangChain处理器参数
- `--chunk-size`: 分块大小（默认：500）
- `--chunk-overlap`: 块重叠大小（默认：50）
- `--max-keywords`: 最大关键词数量（默认：10）
- `--keyword-method`: 关键词提取方法（tfidf/embedding/hybrid，默认：hybrid）
- `--language`: 文本语言（english/chinese，默认：english）

### GLM4.5接口参数
- `--operation`: 操作类型（keyword_extraction/semantic_chunking/summarization/structure_analysis）
- `--api-key`: GLM API密钥（可选，可通过环境变量设置）
- `--base-url`: 自定义API端点（可选）

## 性能优化建议

### 1. 内存管理
- 对于大型PDF文件，建议分批处理
- 使用适当的块大小（通常300-800字符为佳）

### 2. API使用
- GLM4.5 API调用有速率限制，建议合理设置请求间隔
- 可以缓存常用文本的处理结果

### 3. 模型选择
- 根据文本语言选择合适的嵌入模型
- 英文文本推荐：`all-MiniLM-L6-v2`
- 中文文本推荐：`paraphrase-multilingual-MiniLM-L12-v2`

## 故障排除

### 常见问题

1. **Python依赖安装失败**
   ```bash
   # 更新pip
   pip install --upgrade pip
   
   # 使用国内镜像
   pip install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

2. **GLM4.5 API调用失败**
   - 检查API密钥是否正确设置
   - 确认网络连接正常
   - 验证API配额是否充足

3. **PDF解析错误**
   - 确认PDF文件格式正确
   - 检查文件权限
   - 尝试使用其他PDF文件测试

4. **内存不足**
   - 减少块大小
   - 分批处理大型文件
   - 增加系统内存

### 调试模式
```bash
# 启用详细输出
export PYTHONPATH=.:$PYTHONPATH
python python/langchain_processor.py --text "test" --verbose

# 检查C++程序输出
./build/paper-qa your_file.pdf 2>&1 | tee debug.log
```

## 扩展开发

### 添加新的文本处理器
1. 在`python/`目录下创建新的处理器脚本
2. 实现必要的接口方法
3. 在C++代码中添加调用逻辑

### 集成其他LLM模型
1. 修改`python/glm_interface.py`
2. 添加新的模型配置
3. 实现相应的API调用方法

### 自定义分块策略
1. 继承LangChain的TextSplitter类
2. 实现自定义的分块逻辑
3. 在`langchain_processor.py`中集成
