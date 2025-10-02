#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "text_processor.h"
namespace paper_qa {

    class VectorDatabase {
    private:
        std::vector<std::vector<float>> embeddings;
        std::vector<std::string> texts;
        std::vector<TextChunk> chunks;

    public:
        void add_embedding(const std::vector<float>& embedding, const std::string& text,
                           const TextChunk& chunk) {
            embeddings.push_back(embedding);
            texts.push_back(text);
            chunks.push_back(chunk);
        }

        std::vector<std::pair<int, float>> search(const std::vector<float>& query_embedding,
                                                  int top_k = 5) {
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
            if (a.size() != b.size())
                return 0.0f;

            float dot_product = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;

            for (size_t i = 0; i < a.size(); ++i) {
                dot_product += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            if (norm_a == 0.0f || norm_b == 0.0f)
                return 0.0f;

            return dot_product / (sqrt(norm_a) * sqrt(norm_b));
        }
    };
}