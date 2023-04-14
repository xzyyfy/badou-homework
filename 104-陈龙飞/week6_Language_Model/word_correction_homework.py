import copy
import math

import torch
from nnlm_homework import build_vocab, build_model

"""
文本纠错demo
加载同音字字典
加载语言模型
基本原理：
对于文本中每一个字，判断在其同音字中是否有其他字，在替换掉该字时，能使得语言模型计算的成句概率提高
"""


def load_trained_language_model(path):
    char_dim = 128  # 每个字的维度,与训练时保持一直
    window_size = 6  # 样本文本长度,与训练时保持一直
    vocab = build_vocab("vocab.txt")  # 加载字表
    model = build_model(vocab, char_dim)  # 加载模型
    model.load_state_dict(torch.load(path))  # 加载训练好的模型权重
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.window_size = window_size
    model.vocab = vocab
    return model


# 计算文本ppl
def predict(sentence, model):
    prob = 0
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - model.window_size)
            window = sentence[start:i]
            x = [model.vocab.get(char, model.vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = model.vocab.get(target, model.vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0]
            target_prob = pred_prob_distribute[target_index]
            # print(window , "->", target, "prob:", float(target_prob))
            prob += math.log(target_prob, 10)
    return prob


class Corrector:
    def __init__(self, language_model):
        # 语言模型
        self.language_model = language_model
        # 候选字字典
        self.sub_dict = self.load_tongyinzi("tongyin.txt")
        # 成句概率的提升超过阈值则保留修改
        self.threshold = 1

    # 实际上不光是同音字，同形字等也可以加入，本质上是常用的错字
    def load_tongyinzi(self, path):
        tongyinzi_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, tongyin_chars = line.split()
                tongyinzi_dict[char] = list(tongyin_chars)
        return tongyinzi_dict

    # 根据替换字逐句计算成句概率的提升值
    def get_candidate_sentence_prob(self, candidates, char_list, index):
        if not candidates:
            return [-1]
        result = []
        for char in candidates:
            char_list[index] = char
            sentence = "".join(char_list)
            sentence_prob = predict(sentence, self.language_model)
            # 减去基线值，得到提升了多少
            sentence_prob -= self.sentence_prob_baseline
            result.append(sentence_prob)
        return result

    # 纠错逻辑
    def correction(self, string):
        char_list = list(string)
        fix = {}
        # 计算一个原句的成句概率
        self.sentence_prob_baseline = predict(string, self.language_model)
        for index, char in enumerate(char_list):
            candidates = self.sub_dict.get(char, [])
            # 注意使用char_list的拷贝，以免直接修改了原始内容
            candidate_probs = self.get_candidate_sentence_prob(candidates, copy.deepcopy(char_list), index)
            # 如果成句概率的提升大于一定阈值，则记录替换结果
            if max(candidate_probs) > self.threshold:
                # 找到最大成句概率对应的替换字
                sub_char = candidates[candidate_probs.index(max(candidate_probs))]
                print("第%d个字建议修改：%s -> %s, 概率提升： %f" % (index + 1, char, sub_char, max(candidate_probs)))
                fix[index] = sub_char
        # 替换后字符串
        char_list = [fix[i] if i in fix else char for i, char in enumerate(char_list)]
        return "".join(char_list)


lm = load_trained_language_model("财经.pth")

cr = Corrector(lm)
string = "每国货币政册空间不大"
fix_string = cr.correction(string)
print("修改前：", string)
print("修改后：", fix_string)
