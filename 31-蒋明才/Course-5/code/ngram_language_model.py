"""
 #
 # @Author: jmc
 # @Date: 2023/3/19 17:33
 # @Version: v1.0
 # @Description: ngram语言模型，常见n取值为1 2 3
"""
from collections import defaultdict
import math
import jieba
jieba.initialize()


class NGramModel:
    def __init__(self, corpus: list, n: int):
        self.corpus = corpus  # 语料
        self.n = n  # N-gram
        self.sep = "/**/"
        self.ngram_count_dict = dict((i, defaultdict(int)) for i in range(1, self.n + 1))
        self.ngram_prob_dict = dict((i, defaultdict(float)) for i in range(1, self.n + 1))
        self.backward_prob = 0.4
        self.ukn_prob = 1e-5
        self.cal_corpus_ngram_count()
        self.cal_corpus_ngram_prob()

    @staticmethod
    def sentence_segment(sentence: str) -> list:
        return jieba.lcut(sentence)

    # 统计语料库中ngram的数量
    def cal_corpus_ngram_count(self) -> None:
        for _, sentence in enumerate(self.corpus):
            for window in range(1, self.n + 1):
                eles = self.sentence_segment(sentence)
                eles = ["<"] + eles + [">"]
                for idx in range(len(eles) - window + 1):
                    ngram = self.sep.join(eles[idx: idx+window])
                    self.ngram_count_dict[window][ngram] += 1
        return

    # 统计语料库中ngram的概率
    def cal_corpus_ngram_prob(self) -> None:
        gram_1 = sum(self.ngram_count_dict[1].values())  # 1-gram的所有词频之和
        for window in range(1, self.n + 1):
            for ngram, count in self.ngram_count_dict[window].items():
                eles = ngram.split(self.sep)
                if len(eles) == 1:
                    self.ngram_prob_dict[window][ngram] = count / gram_1
                else:
                    prefix = self.sep.join(eles[:-1])
                    prefix_count = self.ngram_count_dict[len(eles) - 1][prefix]
                    self.ngram_prob_dict[window][ngram] = count / prefix_count
        return

    # 获取ngram的概率
    def get_ngram_prob(self, ngram):
        eles = ngram.split(self.sep)
        if ngram in self.ngram_prob_dict[len(eles)]:
            return self.ngram_prob_dict[len(eles)][ngram]
        elif len(eles) == 1:
            return self.ukn_prob
        else:
            curr = self.sep.join(eles[1:])
            return self.backward_prob * self.get_ngram_prob(curr)

    # 预测sentence的概率
    def predict_sentence_prob(self, sentence: str):
        words = self.sentence_segment(sentence)
        words = ["<"] + words + [">"]
        prob = 0
        for idx in range(len(words) - self.n + 1):
            ngram = self.sep.join(words[idx: idx+self.n])
            prob += math.log(math.exp(self.get_ngram_prob(ngram)))
        return prob


if __name__ == '__main__':
    train_corpus = []
    with open("../datasets/财经.txt", encoding="utf-8") as file:
        for line in file.readlines():
            train_corpus.append(line.replace("\n", ""))
    ngm = NGramModel(train_corpus, 3)
    prob = ngm.predict_sentence_prob("没国货币政策空间不大")
    print("当前句子概率", prob)
