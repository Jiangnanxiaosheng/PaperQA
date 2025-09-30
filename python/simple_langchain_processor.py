#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版LangChain文本处理器
不依赖复杂的库，提供基本的文本处理功能
"""

import sys
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
import argparse

# 尝试导入基础库
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    NLTK_AVAILABLE = True
    print("NLTK and sklearn imported successfully")
except ImportError as e:
    print(f"Warning: Could not import NLTK/sklearn: {e}")
    NLTK_AVAILABLE = False

class SimpleLangChainTextProcessor:
    """简化的文本处理器，不依赖复杂的库"""
    
    def __init__(self, language: str = "english"):
        """
        初始化文本处理器
        
        Args:
            language: 文本语言
        """
        self.language = language
        self.stop_words = set()
        self.lemmatizer = None
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化NLP组件"""
        try:
            if NLTK_AVAILABLE:
                # 下载NLTK数据
                print("Downloading NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                print("NLTK data downloaded successfully")
                
                # 初始化词形还原器
                self.lemmatizer = WordNetLemmatizer()
                
                # 加载停用词
                self.stop_words = set(stopwords.words(self.language))
            else:
                # 使用更完整的停用词列表
                if self.language == "english":
                    self.stop_words = {
                        # 基础停用词
                        "the", "a", "an", "and", "or", "but", "if", "because", "as", "until", "while",
                        "of", "at", "by", "for", "with", "about", "against", "between", "into",
                        "through", "during", "before", "after", "above", "below", "to", "from",
                        "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
                        "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
                        "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
                        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
                        "will", "just", "don", "should", "now",
                        # 常见无意义词汇
                        "that", "this", "these", "those", "is", "are", "was", "were", "be", "been",
                        "being", "have", "has", "had", "do", "does", "did", "shall", "will", "would",
                        "could", "should", "may", "might", "must", "i", "you", "he", "she", "it",
                        "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
                        "our", "their", "mine", "yours", "hers", "ours", "theirs", "what", "which",
                        "who", "whom", "whose", "where", "when", "why", "how", "whether", "either",
                        "neither", "each", "every", "any", "some", "all", "both", "half", "several",
                        "many", "much", "few", "little", "more", "most", "less", "least", "enough",
                        "such", "own", "same", "so", "than", "too", "very", "quite", "rather",
                        "somewhat", "however", "nevertheless", "nonetheless", "otherwise", "hence",
                        "therefore", "consequently", "accordingly", "furthermore", "moreover",
                        "besides", "anyway", "anyhow", "still", "yet", "already", "just", "now",
                        "soon", "then", "later", "earlier", "before", "after", "since", "until",
                        "while", "when", "whenever", "wherever", "whereas", "although", "though",
                        "even", "if", "unless", "until", "till", "because", "since", "as", "so",
                        "that", "in", "order", "that", "provided", "that", "assuming", "that",
                        "supposing", "that", "given", "that", "being", "that", "having", "that",
                        "by", "means", "of", "through", "via", "using", "with", "without", "within",
                        "inside", "outside", "upon", "onto", "into", "out", "of", "off", "away",
                        "from", "toward", "towards", "against", "along", "across", "among",
                        "around", "behind", "below", "beneath", "beside", "between", "beyond",
                        "during", "except", "past", "since", "throughout", "under", "underneath",
                        "until", "up", "upon", "with", "within", "without", "aboard", "about",
                        "above", "across", "after", "against", "along", "amid", "among", "around",
                        "as", "at", "before", "behind", "below", "beneath", "beside", "between",
                        "beyond", "but", "by", "down", "during", "except", "for", "from", "in",
                        "inside", "into", "like", "near", "of", "off", "on", "onto", "out",
                        "outside", "over", "past", "plus", "round", "save", "since", "than",
                        "through", "till", "to", "toward", "towards", "under", "underneath",
                        "until", "up", "upon", "with", "within", "without", "ago", "ahead",
                        "apart", "aside", "away", "back", "behind", "down", "forward", "in",
                        "off", "on", "out", "over", "round", "through", "together", "toward",
                        "towards", "under", "up", "aback", "about", "above", "across", "after",
                        "against", "along", "among", "around", "as", "at", "before", "behind",
                        "below", "beneath", "beside", "between", "beyond", "but", "by", "down",
                        "during", "except", "for", "from", "in", "inside", "into", "like",
                        "near", "of", "off", "on", "onto", "out", "outside", "over", "past",
                        "plus", "round", "save", "since", "than", "through", "till", "to",
                        "toward", "towards", "under", "underneath", "until", "up", "upon",
                        "with", "within", "without", "also", "always", "even", "ever", "indeed",
                        "never", "nevertheless", "none", "nonetheless", "no", "not", "n't",
                        "nothing", "nowhere", "nor", "notwithstanding", "only", "otherwise",
                        "rather", "regardless", "save", "similarly", "still", "though", "thus",
                        "too", "unless", "until", "unto", "up", "upon", "whereas", "whether",
                        "while", "yet", "yet", "already", "almost", "approximately", "barely",
                        "completely", "considerably", "deeply", "enough", "entirely", "extremely",
                        "fairly", "fully", "greatly", "hardly", "highly", "how", "however",
                        "incredibly", "indeed", "less", "little", "lots", "more", "most", "much",
                        "nearly", "neither", "never", "not", "notably", "now", "often", "only",
                        "particularly", "quite", "rather", "really", "remarkably", "scarcely",
                        "several", "significantly", "so", "some", "somewhat", "such", "terribly",
                        "that", "too", "totally", "utterly", "very", "well", "see", "case"
                    }
                else:
                    self.stop_words = {
                        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
                        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
                        "自己", "这", "那", "现在", "可以", "但是", "还是", "因为", "什么", "如果"
                    }
                
        except Exception as e:
            print(f"Error initializing NLP components: {e}")
            self.stop_words = set()
            self.lemmatizer = None
    
    def simple_chunking(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        简单的文本分块
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            分块结果列表
        """
        result = []
        start = 0
        
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
        
        print(f"Simple chunking completed with {len(result)} chunks")
        return result
    
    def extract_keywords_simple(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        简化的关键词提取
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        Returns:
            关键词列表
        """
        try:
            if NLTK_AVAILABLE:
                # 使用NLTK进行更精确的处理
                return self._extract_keywords_with_nltk(text, max_keywords)
            else:
                # 使用简单的词频统计
                return self._extract_keywords_by_frequency(text, max_keywords)
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return self._extract_keywords_by_frequency(text, max_keywords)
    
    def _extract_keywords_with_nltk(self, text: str, max_keywords: int) -> List[str]:
        """使用NLTK提取关键词"""
        try:
            # 分词
            words = word_tokenize(text.lower())
            
            # 词形还原和过滤
            filtered_words = []
            for word in words:
                if (word.isalpha() and 
                    word not in self.stop_words and 
                    len(word) > 2):
                    if self.lemmatizer:
                        word = self.lemmatizer.lemmatize(word)
                    filtered_words.append(word)
            
            # 计算词频
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 按频率排序
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # 返回前max_keywords个词
            return [word for word, freq in sorted_words[:max_keywords]]
            
        except Exception as e:
            print(f"Error in NLTK keyword extraction: {e}")
            return self._extract_keywords_by_frequency(text, max_keywords)
    
    def _extract_keywords_by_frequency(self, text: str, max_keywords: int) -> List[str]:
        """使用词频统计提取关键词"""
        # 简单分词（基于空格和标点）
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # 过滤停用词
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # 计算词频
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前max_keywords个词
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def calculate_similarity_simple(self, text1: str, text2: str) -> float:
        """
        简化的相似度计算
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            if NLTK_AVAILABLE:
                # 使用TF-IDF计算相似度
                vectorizer = TfidfVectorizer().fit_transform([text1, text2])
                similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
                return float(similarity)
            else:
                # 使用简单的词汇重叠
                words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
                words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))
                
                if not words1 or not words2:
                    return 0.0
                
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                
                return len(intersection) / len(union)
                
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def process_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50, 
                    max_keywords: int = 10) -> Dict[str, Any]:
        """
        处理文本的完整流程
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            max_keywords: 最大关键词数量
            
        Returns:
            处理结果字典
        """
        result = {
            "chunks": [],
            "keywords": [],
            "similarity_matrix": [],
            "metadata": {
                "total_length": len(text),
                "chunk_count": 0,
                "processing_method": "simple"
            }
        }
        
        try:
            # 文本分块
            chunks = self.simple_chunking(text, chunk_size, chunk_overlap)
            result["chunks"] = chunks
            result["metadata"]["chunk_count"] = len(chunks)
            
            # 提取关键词
            keywords = self.extract_keywords_simple(text, max_keywords)
            result["keywords"] = keywords
            
            # 计算块间相似度
            if len(chunks) > 1:
                similarity_matrix = []
                for i in range(len(chunks)):
                    row = []
                    for j in range(len(chunks)):
                        if i == j:
                            row.append(1.0)
                        else:
                            similarity = self.calculate_similarity_simple(chunks[i]["text"], chunks[j]["text"])
                            row.append(similarity)
                    similarity_matrix.append(row)
                result["similarity_matrix"] = similarity_matrix
            
        except Exception as e:
            print(f"Error processing text: {e}")
            result["error"] = str(e)
        
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Simple LangChain Text Processor")
    parser.add_argument("--text", type=str, help="Input text to process")
    parser.add_argument("--file", type=str, help="Input file containing text to process")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--max-keywords", type=int, default=10, help="Maximum keywords")
    parser.add_argument("--language", type=str, default="english", help="Text language")
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
    
    # 创建处理器
    processor = SimpleLangChainTextProcessor(language=args.language)
    
    # 处理文本
    result = processor.process_text(
        text=text,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
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