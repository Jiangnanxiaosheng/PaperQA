#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GLM4.5模型接口
用于与GLM4.5模型交互，提供高级文本理解能力
"""

import sys
import json
import os
import re
import argparse
from typing import Dict, List, Any, Optional, Union
import requests
import time
from datetime import datetime

class GLMInterface:
    """GLM4.5模型接口类"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化GLM接口
        
        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        self.base_url = base_url or os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        
        if not self.api_key:
            print("Warning: GLM API key not provided. Some functionality may be limited.")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # 模型配置
        self.model_name = "glm-4"  # GLM4.5的模型名称
        self.max_tokens = 2000
        self.temperature = 0.7
        self.top_p = 0.9
        
        # 请求限制
        self.rate_limit_delay = 1.0  # 请求间隔（秒）
        self.last_request_time = 0
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送API请求
        
        Args:
            endpoint: API端点
            payload: 请求负载
            
        Returns:
            响应数据
        """
        # 速率限制
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
        
        self.last_request_time = time.time()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {"error": str(e)}
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        聊天完成接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            响应数据
        """
        payload = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stream": False
        }
        
        return self._make_request("chat/completions", payload)
    
    def extract_keywords_advanced(self, text: str, max_keywords: int = 10, 
                                domain: Optional[str] = None) -> List[str]:
        """
        使用GLM4.5提取高级关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            domain: 专业领域（可选）
            
        Returns:
            关键词列表
        """
        domain_prompt = f" in the field of {domain}" if domain else ""
        
        prompt = f"""
        Please extract the most important keywords and key phrases from the following text{domain_prompt}.
        Consider semantic importance, technical terms, and conceptual significance.
        
        Requirements:
        1. Extract exactly {max_keywords} most important keywords/phrases
        2. Focus on substantive content, not common words
        3. Include both single words and meaningful phrases
        4. Consider the technical and academic context
        5. Return only the keywords/phrases, one per line
        
        Text:
        {text[:3000]}  # 限制文本长度
        
        Keywords:
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in keyword extraction and semantic analysis."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages)
        
        if "error" in response:
            print(f"Error in keyword extraction: {response['error']}")
            return []
        
        try:
            content = response["choices"][0]["message"]["content"]
            keywords = [line.strip() for line in content.split('\n') if line.strip()]
            return keywords[:max_keywords]
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return []
    
    def semantic_chunking_advanced(self, text: str, chunk_size: int = 500, 
                                  chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        使用GLM4.5进行高级语义分块
        
        Args:
            text: 输入文本
            chunk_size: 目标块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            分块结果
        """
        prompt = f"""
        Please divide the following text into meaningful chunks for a RAG (Retrieval-Augmented Generation) system.
        
        Requirements:
        1. Each chunk should be around {chunk_size} characters (can vary slightly for semantic completeness)
        2. Maintain semantic coherence within each chunk
        3. Ensure logical flow between chunks
        4. Include {chunk_overlap} characters of overlap between adjacent chunks when appropriate
        5. Preserve important context and relationships
        6. Return the result as a JSON array of objects with 'text' and 'summary' fields
        
        Text:
        {text[:4000]}  # 限制文本长度
        
        Response format:
        [
            {
                "text": "Chunk text content...",
                "summary": "Brief summary of this chunk"
            },
            ...
        ]
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in text segmentation and semantic analysis for RAG systems."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages)
        
        if "error" in response:
            print(f"Error in semantic chunking: {response['error']}")
            return []
        
        try:
            content = response["choices"][0]["message"]["content"]
            chunks = json.loads(content)
            
            # 添加元数据
            for i, chunk in enumerate(chunks):
                chunk["chunk_id"] = i
                chunk["metadata"] = {
                    "method": "glm-advanced",
                    "created_at": datetime.now().isoformat()
                }
            
            return chunks
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return []
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        生成文本摘要
        
        Args:
            text: 输入文本
            max_length: 摘要最大长度
            
        Returns:
            摘要文本
        """
        prompt = f"""
        Please provide a concise summary of the following text in no more than {max_length} characters.
        
        Text:
        {text[:2000]}
        
        Summary:
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in text summarization."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages)
        
        if "error" in response:
            print(f"Error in summarization: {response['error']}")
            return ""
        
        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return ""
    
    def answer_question(self, question: str, context: str) -> str:
        """
        基于给定上下文回答问题
        
        Args:
            question: 问题
            context: 上下文文本
            
        Returns:
            答案
        """
        prompt = f"""
        Based on the following context, please answer the question accurately and concisely.
        
        Context:
        {context[:3000]}
        
        Question:
        {question}
        
        Answer:
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages)
        
        if "error" in response:
            print(f"Error in question answering: {response['error']}")
            return ""
        
        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return ""
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        分析文本结构
        
        Args:
            text: 输入文本
            
        Returns:
            结构分析结果
        """
        prompt = f"""
        Please analyze the structure of the following text and provide a detailed breakdown.
        
        Analysis should include:
        1. Main topics and themes
        2. Logical sections and their relationships
        3. Key arguments and supporting evidence
        4. Technical terms and concepts
        5. Overall structure type (e.g., academic paper, technical report, etc.)
        
        Return the analysis as a JSON object with appropriate fields.
        
        Text:
        {text[:3000]}
        
        Analysis (JSON format):
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in text structure analysis."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages)
        
        if "error" in response:
            print(f"Error in structure analysis: {response['error']}")
            return {}
        
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return {}
    
    def process_text_with_glm(self, text: str, operations: List[str], 
                             chunk_size: int = 500, max_keywords: int = 10) -> Dict[str, Any]:
        """
        使用GLM4.5处理文本的完整流程
        
        Args:
            text: 输入文本
            operations: 要执行的操作列表
            chunk_size: 块大小
            max_keywords: 最大关键词数量
            
        Returns:
            处理结果
        """
        result = {
            "original_text_length": len(text),
            "operations_performed": [],
            "results": {},
            "metadata": {
                "model": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        for operation in operations:
            try:
                if operation == "keyword_extraction":
                    keywords = self.extract_keywords_advanced(text, max_keywords)
                    result["results"]["keywords"] = keywords
                    result["operations_performed"].append("keyword_extraction")
                
                elif operation == "semantic_chunking":
                    chunks = self.semantic_chunking_advanced(text, chunk_size)
                    result["results"]["chunks"] = chunks
                    result["operations_performed"].append("semantic_chunking")
                
                elif operation == "summarization":
                    summary = self.generate_summary(text)
                    result["results"]["summary"] = summary
                    result["operations_performed"].append("summarization")
                
                elif operation == "structure_analysis":
                    structure = self.analyze_text_structure(text)
                    result["results"]["structure"] = structure
                    result["operations_performed"].append("structure_analysis")
                
            except Exception as e:
                print(f"Error in operation {operation}: {e}")
                result["results"][operation] = {"error": str(e)}
        
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GLM4.5 Interface")
    parser.add_argument("--text", type=str, help="Input text to process")
    parser.add_argument("--file", type=str, help="Input file containing text to process")
    parser.add_argument("--operation", type=str, nargs="+", 
                       choices=["keyword_extraction", "semantic_chunking", "summarization", "structure_analysis"],
                       default=["keyword_extraction"], help="Operations to perform")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for chunking")
    parser.add_argument("--max-keywords", type=int, default=10, help="Maximum keywords to extract")
    parser.add_argument("--api-key", type=str, help="GLM API key")
    parser.add_argument("--base-url", type=str, help="GLM API base URL")
    parser.add_argument("--output", type=str, help="Output file (JSON format)")
    
    args = parser.parse_args()
    
    # 获取输入文本
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        print("Error: Either --text or --file must be provided")
        sys.exit(1)
    
    # 创建GLM接口
    glm_interface = GLMInterface(api_key=args.api_key, base_url=args.base_url)
    
    # 处理文本
    result = glm_interface.process_text_with_glm(
        text=text,
        operations=args.operation,
        chunk_size=args.chunk_size,
        max_keywords=args.max_keywords
    )
    
    # 输出结果
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()