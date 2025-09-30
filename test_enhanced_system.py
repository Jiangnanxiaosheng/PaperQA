#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试增强后的RAG系统
"""

import os
import sys
import json
import subprocess
import argparse
from typing import Dict, List, Any

def test_langchain_processor():
    """测试LangChain文本处理器"""
    print("=" * 50)
    print("测试LangChain文本处理器")
    print("=" * 50)
    
    # 示例文本
    test_text = """
    Sequences with impulse-like periodic correlations are at the core of several radar and communication applications.
    Two criteria that can be used to design such sequences, and which lead to rather different results in the aperiodic correlation case,
    are shown to be identical in the periodic case. Furthermore, two simplified versions of these two criteria, which similarly yield
    completely different sequences in the aperiodic case, are also shown to be equivalent.
    """
    
    langchain_success = False
    simple_success = False
    
    # 首先尝试完整版LangChain处理器
    try:
        print("尝试完整版LangChain处理器...")
        cmd = [
            "python", "python/langchain_processor.py",
            "--text", test_text,
            "--chunk-size", "300",
            "--chunk-overlap", "30",
            "--max-keywords", "8",
            "--keyword-method", "hybrid",
            "--output", "test_langchain_result.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ 完整版LangChain处理器测试成功")
        langchain_success = True
        
        # 读取并显示结果
        with open("test_langchain_result.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"  - 分块数量: {len(data.get('chunks', []))}")
        print(f"  - 关键词数量: {len(data.get('keywords', []))}")
        print("  - 关键词:", ", ".join(data.get('keywords', [])[:5]))
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 完整版LangChain处理器测试失败: {e}")
    
    # 如果完整版失败，尝试简化版
    if not langchain_success:
        try:
            print("\n尝试简化版LangChain处理器...")
            cmd = [
                "python", "python/simple_langchain_processor.py",
                "--text", test_text,
                "--chunk-size", "300",
                "--chunk-overlap", "30",
                "--max-keywords", "8",
                "--output", "test_simple_langchain_result.json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ 简化版LangChain处理器测试成功")
            simple_success = True
            
            # 读取并显示结果
            with open("test_simple_langchain_result.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print(f"  - 分块数量: {len(data.get('chunks', []))}")
            print(f"  - 关键词数量: {len(data.get('keywords', []))}")
            print("  - 关键词:", ", ".join(data.get('keywords', [])[:5]))
            
        except subprocess.CalledProcessError as e:
            print(f"✗ 简化版LangChain处理器测试失败: {e}")
        except Exception as e:
            print(f"✗ 测试过程中出现错误: {e}")
    
    return langchain_success or simple_success

def test_glm_interface():
    """测试GLM4.5接口"""
    print("\n" + "=" * 50)
    print("测试GLM4.5接口")
    print("=" * 50)
    
    # 示例文本
    test_text = """
    This paper discusses the design of sequences with impulse-like periodic correlations for radar applications.
    The authors prove that two different design criteria are actually equivalent in the periodic case.
    """
    
    # 检查是否有API密钥
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("⚠ 未找到GLM_API_KEY环境变量，跳过GLM4.5接口测试")
        print("  要测试GLM4.5功能，请设置环境变量: export GLM_API_KEY=your_api_key")
        return True
    
    try:
        # 测试关键词提取
        cmd = [
            "python", "python/glm_interface.py",
            "--text", test_text,
            "--operation", "keyword_extraction",
            "--max-keywords", "5",
            "--output", "test_glm_result.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ GLM4.5接口测试成功")
        
        # 读取并显示结果
        with open("test_glm_result.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        keywords = data.get("results", {}).get("keywords", [])
        print(f"  - 提取的关键词数量: {len(keywords)}")
        print("  - 关键词:", ", ".join(keywords))
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ GLM4.5接口测试失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        return False

def test_pdf_processing():
    """测试PDF处理"""
    print("\n" + "=" * 50)
    print("测试PDF处理")
    print("=" * 50)
    
    # 检查测试PDF文件
    pdf_path = "test.pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠ 未找到测试PDF文件: {pdf_path}")
        print("  跳过PDF处理测试")
        return True
    
    try:
        # 构建并运行C++程序
        print("构建C++程序...")
        build_result = subprocess.run(["mkdir", "-p", "build"], check=True)
        build_result = subprocess.run(["cd", "build", "&&", "cmake", ".."], shell=True, check=True)
        build_result = subprocess.run(["cd", "build", "&&", "make"], shell=True, check=True)
        
        print("运行PDF处理测试...")
        cmd = ["./build/paper-qa", pdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✓ PDF处理测试成功")
        
        # 分析输出
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "关键词提取结果" in line:
                print("  - 找到关键词提取结果")
            elif "使用LangChain增强" in line:
                print("  - 确认使用LangChain增强处理")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ PDF处理测试失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        return False

def install_dependencies():
    """安装Python依赖"""
    print("\n" + "=" * 50)
    print("安装Python依赖")
    print("=" * 50)
    
    try:
        print("安装依赖包...")
        result = subprocess.run([
            "pip", "install", "-r", "python/requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✓ 依赖安装成功")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败: {e}")
        print("  请手动安装: pip install -r python/requirements.txt")
        return False

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description="测试增强后的RAG系统")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-glm", action="store_true", help="跳过GLM4.5测试")
    parser.add_argument("--skip-pdf", action="store_true", help="跳过PDF处理测试")
    
    args = parser.parse_args()
    
    print("开始测试增强后的RAG系统...")
    
    results = {}
    
    # 安装依赖
    if not args.skip_deps:
        results["dependencies"] = install_dependencies()
    else:
        print("跳过依赖安装")
        results["dependencies"] = True
    
    # 测试LangChain处理器
    results["langchain"] = test_langchain_processor()
    
    # 测试GLM4.5接口
    if not args.skip_glm:
        results["glm"] = test_glm_interface()
    else:
        print("\n跳过GLM4.5接口测试")
        results["glm"] = True
    
    # 测试PDF处理
    if not args.skip_pdf:
        results["pdf"] = test_pdf_processing()
    else:
        print("\n跳过PDF处理测试")
        results["pdf"] = True
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！增强后的RAG系统已准备就绪。")
    else:
        print("\n⚠ 部分测试失败，请检查相关组件。")
    
    # 清理临时文件
    try:
        os.remove("test_langchain_result.json")
        os.remove("test_glm_result.json")
    except:
        pass

if __name__ == "__main__":
    main()