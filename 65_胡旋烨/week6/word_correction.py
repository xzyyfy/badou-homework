import copy

import torch

from ngram_language_model import NgramLanguageModel
from n_gram_pytorch import NGramModel, clean, build_vocab

"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""


class Corrector:
    def __init__(self, language_model):
        # 语言模型
        self.language_model = language_model
        # 候选字字典
        self.sub_dict = self.load_easy_wrong("tongyin.txt")
        # 成句概率的提升超过阈值则保留修改
        self.threshold = 7

    @staticmethod
    def load_easy_wrong(path):
        # 实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
        wrong_dict = {}
        with open(path, encoding="utf8") as wrongs:
            for line in wrongs:
                char, wrong_chars = line.split()
                wrong_dict[char] = list(wrong_chars)
        return wrong_dict

    # 纠错逻辑
    def correction(self, ss):
        # 1,先记录一下这个句子的成句概率
        score = self.language_model.predict(ss)
        words = list(ss)

        for i, word in enumerate(words):
            cp_words = copy.deepcopy(words)

            similars = self.sub_dict.get(word, [])
            if not similars:
                continue
            for similar in similars:
                cp_words[i] = similar
                cp_string = ''.join(cp_words)
                cp_score = self.language_model.predict(cp_string)
                if cp_score - score >= self.threshold:
                    # 如果超过阈值，则调整文字和评分
                    words = copy.deepcopy(cp_words)
                    score = copy.deepcopy(cp_score)
        return "".join(words), score


if __name__ == '__main__':
    with open("财经.txt", encoding="utf8") as f:
        corpus = f.readlines()
    corpus = [line.strip() for line in corpus]

    # 使用深度学习算法的用这几行代码——效果不好
    # corpus = clean(corpus)
    # vocab = build_vocab(corpus)
    # lm = NGramModel(vocab)
    # lm.load_state_dict(torch.load("n_gram_model.pth"))
    # lm.eval()

    # 使用概率算法的用这一行代码——效果好
    lm = NgramLanguageModel(corpus, 3)

    cr = Corrector(lm)
    string = "每国货币政册空间不大"  # 美国货币政策空间不大
    fix_string, _ = cr.correction(string)
    print("修改前：", string)
    print("修改后：", fix_string)
