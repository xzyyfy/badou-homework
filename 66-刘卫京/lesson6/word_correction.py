# -- coding: utf-8 --
# @Time : 2023/3/20 15:56
# @Author : liuweijing
# @File : word_correction.py

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
    def correction_1(self, string):
        word_list = list(string)
        correct_list = list(string)
        correct_string = string
        for index, word in enumerate(word_list):
            # 整句计算ppl
            org_ppl = self.language_model.calc_sentence_ppl(correct_string)
            # 从字典中找到所有的同音字
            words = self.sub_dict.get(word)
            if not words:
                # 如果没有在同音字典中找到这个字，就把原字返回
                correct_list.append(words)
                continue
            # 计算每个同音字的成句概率，找到最大的一个
            for idx, each_word in enumerate(words):
                # 遍历所有同音字 拼接到整句上
                word_list = correct_list[0: index] + [each_word] + correct_list[index + 1:]
                
                new_string = "".join(word_list)
                # 计算拼接后句子的ppl
                sentence_ppl = self.language_model.calc_sentence_ppl(new_string)
                if sentence_ppl < org_ppl:
                    org_ppl = sentence_ppl
                    # ppl低表明成句概率更高，把当前字改成ppl最低的句子
                    correct_list[index] = each_word
            correct_string = "".join(correct_list)
        return correct_string

corpus = open("财经.txt", encoding="utf8").readlines()
lm = NgramLanguageModel(corpus, 3)

cr = Corrector(lm)
string = "每国货币正册空间不搭"  # 美国货币政策空间不大
fix_string = cr.correction_1(string)
print("修改前：", string)
print("修改后：", fix_string)
