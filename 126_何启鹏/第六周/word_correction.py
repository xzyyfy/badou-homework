import json
import copy
from ngram_language_model import NgramLanguageModel

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
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        # 成句概率的提升超过阈值则保留修改
        self.threshold = 7

    # 实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        tongyinzi_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    # 纠错逻辑
    def correction(self, string):
        # 将输入字符串转为列表
        chars = list(string)
        # 遍历每个字，生成修改方案
        candidates = []
        for i, char in enumerate(chars):
            # 如果该字存在同音字
            if char in self.sub_dict:
                # 生成所有可替换的候选字列表
                subs = self.sub_dict[char]
                for sub in subs:
                    # 复制原字符列表，替换当前字
                    new_chars = copy.deepcopy(chars)
                    new_chars[i] = sub
                    # 计算修改前后的成句概率s
                    original_prob = self.language_model.predict(chars)
                    new_prob = self.language_model.predict(new_chars)
                    # 如果修改后的概率比原始概率高且提升超过阈值，则加入候选列表，并且修改原句
                    if new_prob > original_prob and new_prob - original_prob > self.threshold:
                        candidates.append((''.join(new_chars), new_prob))
                        chars = new_chars
        # 如果没有候选，返回原始字符串
        if not candidates:
            return string
        # 返回概率最高的候选
        return max(candidates, key=lambda x: x[1])[0]


corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, 3)

cr = Corrector(lm)
string = "每国货币政册空间不大"  # 美国货币政策空间不大
string1 = "美国国货币政册空间不大"  # -54.9513157467608
string2 = "每国货币政策空间不大"  # -37.54748070543113
string3 = "美国货币政策空间不大"  # -29.345626277019186
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string)
