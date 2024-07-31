from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDF:
    def __init__(self, corpus):
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer()

    def get_similarity(self, sentence1, sentence2):
        # 将句子转换为TF-IDF向量
        tfidf_matrix = self.vectorizer.transform([sentence1, sentence2])
        # 计算两个句子的相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        # 返回相似度
        return similarity[0][0]
