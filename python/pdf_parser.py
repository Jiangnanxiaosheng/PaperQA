#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF解析脚本
使用PyPDF2库提取PDF文本内容
"""

import sys
import json
import os
from typing import Dict, List, Any

try:
    import PyPDF2
except ImportError:
    print("Error: PyPDF2 library not found. Please install it using: pip install PyPDF2")
    sys.exit(1)

def extract_text_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    从PDF文件中提取文本内容和元数据
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        包含文本内容和元数据的字典
    """
    result = {
        "text": "",
        "pages": [],
        "metadata": {},
        "page_count": 0
    }
    
    try:
        # 打开PDF文件
        with open(pdf_path, 'rb') as file:
            # 创建PDF阅读器对象
            pdf_reader = PyPDF2.PdfReader(file)
            
            # 获取页数
            page_count = len(pdf_reader.pages)
            result["page_count"] = page_count
            
            # 提取元数据
            if pdf_reader.metadata:
                metadata = pdf_reader.metadata
                result["metadata"] = {
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                    "subject": metadata.get("/Subject", ""),
                    "creator": metadata.get("/Creator", ""),
                    "producer": metadata.get("/Producer", ""),
                    "creation_date": str(metadata.get("/CreationDate", "")),
                    "modification_date": str(metadata.get("/ModDate", ""))
                }
            
            # 提取每一页的文本
            all_text = ""
            for page_num in range(page_count):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # 清理文本
                page_text = clean_text(page_text)
                
                # 添加到结果中
                result["pages"].append(page_text)
                all_text += page_text + "\n\n"
            
            result["text"] = all_text.strip()
            
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return {}
    
    return result

def clean_text(text: str) -> str:
    """
    清理提取的文本
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 替换多个空格为单个空格
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符
    text = text.replace('\x00', '')
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Usage: python pdf_parser.py <pdf_file_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # 提取文本内容
    result = extract_text_from_pdf(pdf_path)
    
    if not result:
        print("Error: Failed to extract text from PDF")
        sys.exit(1)
    
    # 输出结果（JSON格式）
    # 使用JSON格式以便C++端解析
    import json
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()