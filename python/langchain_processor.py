#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangChain文本处理器
使用LangChain和现代NLP技术进行智能文本处理
"""

import sys
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
import argparse

# 首先导入基础库
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 尝试导入LangChain相关库
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
    print("LangChain imported successfully")
except ImportError as e:
    print(f"Warning: Could not import LangChain components: {e}")
    LANGCHAIN_AVAILABLE = False

# 尝试导入SemanticChunker（可能在某些版本中不可用）
try:
    from langchain.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
    print("SemanticChunker imported successfully")
except ImportError:
    print("Warning: SemanticChunker not available, will use fallback method")
    SEMANTIC_CHUNKER_AVAILABLE = False

# 尝试导入其他可选库
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("SentenceTransformer imported successfully")
except ImportError:
    print("Warning: SentenceTransformer not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    print("spaCy imported successfully")
except ImportError:
    print("Warning: spaCy not available")
    SPACY_AVAILABLE = False

class LangChainTextProcessor:
    """使用LangChain的智能文本处理器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", language: str = "english"):
        """
        初始化文本处理器
        
        Args:
            model_name: 使用的嵌入模型名称
            language: 文本语言
        """
        self.model_name = model_name
        self.language = language
        self.embeddings = None
        self.nlp = None
        self.lemmatizer = None
        self.stop_words = set()
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化NLP组件"""
        try:
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
            
            # 初始化嵌入模型（如果LangChain可用）
            if LANGCHAIN_AVAILABLE:
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=self.model_name,
                        model_kwargs={'device': 'cpu'}
                    )
                    print("HuggingFaceEmbeddings initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize HuggingFaceEmbeddings: {e}")
                    self.embeddings = None
            else:
                self.embeddings = None
            
            # 初始化spaCy模型（如果可用）
            if SPACY_AVAILABLE:
                try:
                    if self.language == "english":
                        self.nlp = spacy.load("en_core_web_sm")
                    else:
                        self.nlp = spacy.load("zh_core_web_sm")
                    print("spaCy model loaded successfully")
                except OSError:
                    print(f"Warning: spaCy model for {self.language} not found. Using basic processing.")
                    self.nlp = None
            else:
                self.nlp = None
            
        except Exception as e:
            print(f"Error initializing NLP components: {e}")
            # 使用基本处理作为后备
            self.embeddings = None
            self.nlp = None
            self.lemmatizer = None
    
    def semantic_chunking(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        使用语义分块进行文本分割
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            分块结果列表
        """
        try:
            # 检查是否可以使用语义分块
            if self.embeddings is None or not SEMANTIC_CHUNKER_AVAILABLE or not LANGCHAIN_AVAILABLE:
                print("Using basic chunking (semantic chunking not available)")
                return self._basic_chunking(text, chunk_size, chunk_overlap)
            
            # 使用语义分块器
            text_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
                number_of_chunks=None
            )
            
            # 创建文档
            doc = Document(page_content=text)
            
            # 分块
            chunks = text_splitter.split_documents([doc])
            
            # 转换为结果格式
            result = []
            for i, chunk in enumerate(chunks):
                result.append({
                    "text": chunk.page_content,
                    "chunk_id": i,
                    "metadata": chunk.metadata or {}
                })
            
            print(f"Semantic chunking completed with {len(result)} chunks")
            return result
            
        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            print("Falling back to basic chunking")
            return self._basic_chunking(text, chunk_size, chunk_overlap)
    
    def _basic_chunking(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        基本文本分块（后备方案）
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            分块结果列表
        """
        # 如果LangChain可用，使用其分块器
        if LANGCHAIN_AVAILABLE:
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                chunks = text_splitter.split_text(text)
                
                result = []
                for i, chunk in enumerate(chunks):
                    result.append({
                        "text": chunk,
                        "chunk_id": i,
                        "metadata": {"method": "recursive"}
                    })
                
                print(f"Basic chunking completed with {len(result)} chunks")
                return result
            except Exception as e:
                print(f"Error using LangChain chunker: {e}")
        
        # 回退到简单的分块方法
        print("Using fallback simple chunking method")
        result = []
        start = 0
        
        while start < len(text):
            # 计算块结束位置
            end = start + chunk_size
            
            # 如果不是最后一块，尝试在句子边界处分割
            if end < len(text):
                # 寻找最近的句子结束符
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?\n':
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
        
        print(f"Fallback chunking completed with {len(result)} chunks")
        return result
    
    def extract_keywords(self, text: str, max_keywords: int = 10, method: str = "hybrid") -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            method: 提取方法 ("tfidf", "embedding", "hybrid")
            
        Returns:
            关键词列表
        """
        try:
            if method == "tfidf":
                return self._extract_keywords_tfidf(text, max_keywords)
            elif method == "embedding":
                return self._extract_keywords_embedding(text, max_keywords)
            else:  # hybrid
                return self._extract_keywords_hybrid(text, max_keywords)
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return self._extract_keywords_fallback(text, max_keywords)
    
    def _extract_keywords_tfidf(self, text: str, max_keywords: int) -> List[str]:
        """使用TF-IDF提取关键词"""
        # 预处理文本
        sentences = sent_tokenize(text)
        
        # 计算TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_keywords * 2,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # 获取最重要的词
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        top_indices = tfidf_scores.argsort()[-max_keywords:][::-1]
        
        keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
        
        return keywords[:max_keywords]
    
    def _extract_keywords_embedding(self, text: str, max_keywords: int) -> List[str]:
        """使用嵌入相似度提取关键词"""
        if self.embeddings is None:
            return self._extract_keywords_tfidf(text, max_keywords)
        
        # 分词
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        # 去重
        unique_words = list(set(words))
        
        if len(unique_words) <= max_keywords:
            return unique_words
        
        # 计算词嵌入
        try:
            word_embeddings = self.embeddings.embed_documents(unique_words)
            text_embedding = self.embeddings.embed_query(text)
            
            # 计算相似度
            similarities = cosine_similarity([text_embedding], word_embeddings)[0]
            
            # 获取最相似的词
            top_indices = similarities.argsort()[-max_keywords:][::-1]
            keywords = [unique_words[i] for i in top_indices]
            
            return keywords
        except Exception as e:
            print(f"Error in embedding-based keyword extraction: {e}")
            return self._extract_keywords_tfidf(text, max_keywords)
    
    def _extract_keywords_hybrid(self, text: str, max_keywords: int) -> List[str]:
        """混合方法提取关键词"""
        # 获取TF-IDF关键词
        tfidf_keywords = self._extract_keywords_tfidf(text, max_keywords)
        
        # 获取嵌入关键词
        embedding_keywords = self._extract_keywords_embedding(text, max_keywords)
        
        # 合并和去重
        all_keywords = list(set(tfidf_keywords + embedding_keywords))
        
        if len(all_keywords) <= max_keywords:
            return all_keywords
        
        # 如果关键词太多，使用TF-IDF分数进行排序
        try:
            vectorizer = TfidfVectorizer(vocabulary=all_keywords)
            tfidf_matrix = vectorizer.fit_transform([text])
            scores = tfidf_matrix.toarray()[0]
            
            # 排序并选择前max_keywords个
            keyword_scores = list(zip(all_keywords, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw[0] for kw in keyword_scores[:max_keywords]]
        except Exception as e:
            print(f"Error in hybrid keyword extraction: {e}")
            return all_keywords[:max_keywords]
    
    def _extract_keywords_fallback(self, text: str, max_keywords: int) -> List[str]:
        """后备关键词提取方法"""
        words = word_tokenize(text.lower())
        words = [word for word in words 
                if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        # 计算词频
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            if self.embeddings is None:
                # 回退到TF-IDF相似度
                return self._calculate_tfidf_similarity(text1, text2)
            
            # 使用嵌入计算相似度
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return self._calculate_tfidf_similarity(text1, text2)
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """使用TF-IDF计算相似度"""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def process_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50, 
                    max_keywords: int = 10, keyword_method: str = "hybrid") -> Dict[str, Any]:
        """
        处理文本的完整流程
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            max_keywords: 最大关键词数量
            keyword_method: 关键词提取方法
            
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
                "processing_method": "langchain"
            }
        }
        
        try:
            # 文本分块
            chunks = self.semantic_chunking(text, chunk_size, chunk_overlap)
            result["chunks"] = chunks
            result["metadata"]["chunk_count"] = len(chunks)
            
            # 提取关键词
            keywords = self.extract_keywords(text, max_keywords, keyword_method)
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
                            similarity = self.calculate_similarity(chunks[i]["text"], chunks[j]["text"])
                            row.append(similarity)
                    similarity_matrix.append(row)
                result["similarity_matrix"] = similarity_matrix
            
        except Exception as e:
            print(f"Error processing text: {e}")
            result["error"] = str(e)
        
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LangChain Text Processor")
    parser.add_argument("--text", type=str, help="Input text to process")
    parser.add_argument("--file", type=str, help="Input file containing text to process")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--max-keywords", type=int, default=10, help="Maximum keywords")
    parser.add_argument("--keyword-method", type=str, default="hybrid", 
                       choices=["tfidf", "embedding", "hybrid"], help="Keyword extraction method")
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
    processor = LangChainTextProcessor(language=args.language)
    
    # 处理文本
    result = processor.process_text(
        text=text,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_keywords=args.max_keywords,
        keyword_method=args.keyword_method
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