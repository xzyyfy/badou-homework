import json
import copy
from ngram_language_model import NgramLanguageModel

'''
使用语言模型做文本纠错。
'''

class Correction:
    def __init__(self, language_model):
        self.language_model = language_model
        self.tongyin_dict = self.load_dict("week6 语言模型和预训练\\tongyin.txt")
        self.threshold = 7

    def load_dict(self, path):
        corection_dict = {}
        with open(path, encoding="utf8") as f:
            for line in f:
                char, chars = line.split()
                corection_dict[char] = list(chars)
        return corection_dict
    
    def get_sentence_p_candidates(self, candidates, char_list, index):
        if candidates == []:
            return [-1]
        result = []
        for char in candidates:
            char_list[index] = char
            new_sentence = "".join(char_list)
            sentence_p = self.language_model.predict(new_sentence)
            sentence_p -= self.sentence_p_baseline
            result.append(sentence_p)
        return result

    def correction(self, sentence):
        char_list = list(sentence)
        fix_list = {}
        self.sentence_p_baseline = self.language_model.predict(sentence)
        for index, char in enumerate(char_list):
            candidates = self.tongyin_dict.get(char,[])
            candidate_p = self.get_sentence_p_candidates(candidates, copy.deepcopy(char_list),index)
            if max(candidate_p) > self.threshold:
                new_char = candidates[candidate_p.index(max(candidate_p))]
                fix_list[index] = new_char
        char_list = [fix_list[i] if i in fix_list else char for i, char in enumerate(char_list)]
        return "".join(char_list)


corpus = open("week6 语言模型和预训练\时尚.txt", encoding="utf8").readlines()
model = NgramLanguageModel(corpus, 3)

correct = Correction(model)
string = "画草茶有益剪康"
fix_string = correct.correction(string)
print("修改前：", string)
print("修改后：",fix_string)