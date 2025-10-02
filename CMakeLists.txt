cmake_minimum_required(VERSION 3.16)  # FetchContent需要3.16或更高版本
project(PaperQASystem VERSION 1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 包含目录
include_directories(include)

# 源文件
set(SOURCES
    src/main.cpp
    src/pdf_parser.cpp
    src/text_processor.cpp
    # src/vectorizer.cpp
    #src/vector_db.cpp
    # src/llm_client.cpp
    # src/rag_engine.cpp
    # src/web_server.cpp
)

# 可执行文件
add_executable(paper-qa ${SOURCES})

# --- 新增部分 ---
# 1. 创建一个目标，将 python 目录下的所有文件复制到构建目录的 python 子目录中
add_custom_target(copy_python_scripts ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/python
            ${CMAKE_BINARY_DIR}/python
    DEPENDS ${CMAKE_SOURCE_DIR}/python/pdf_parser.py  ${CMAKE_SOURCE_DIR}/python/glm_interface.py ${CMAKE_SOURCE_DIR}/python/requirements.txt
)

# 2. 让可执行文件依赖于这个复制目标
# 这样，每次构建 paper-qa 时，都会先确保脚本已经被复制过去
add_dependencies(paper-qa copy_python_scripts)

# 3. 添加安装Python依赖的目标（可选）
add_custom_target(install_python_deps
    COMMAND ${CMAKE_COMMAND} -E echo "Installing Python dependencies..."
    COMMAND pip install -r ${CMAKE_BINARY_DIR}/python/requirements.txt
    COMMENT "Installing Python dependencies for GLM and text processing"
    VERBATIM
)

# 4. 添加一个便捷目标来完整设置Python环境
add_custom_target(setup_python_env
    DEPENDS copy_python_scripts install_python_deps
    COMMENT "Setting up Python environment with all dependencies"
)
# --- 新增部分结束 ---

# 添加测试子目录（可选）
option(ENABLE_TESTS "Enable building tests" OFF)
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

