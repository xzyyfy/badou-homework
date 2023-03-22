"""
 #
 # @Author: jmc
 # @Date: 2023/3/20 9:22
 # @Version: v1.0
 # @Description: 句子纠错
"""
from ngram_language_model import NGramModel
from copy import deepcopy
from typing import Union, Dict, List


class SentenceCorrect:
    def __init__(self, synonyms_dict: Dict[str, List[int]], ngram_model: NGramModel, threshold=0.2):
        self.synonyms_dict = synonyms_dict
        self.ngram_model = ngram_model
        self.threshold = threshold

    # 计算替换后句子的最大概率
    def cal_replace_max_prob(self, rep_idx: int, sentence: str) -> Union[int, tuple]:
        rep_word = sentence[rep_idx]
        if rep_word not in self.synonyms_dict:
            return -1
        synonyms = self.synonyms_dict[rep_word]
        sentence_lst = list(sentence)
        probs = {}
        for word in synonyms:
            new_sentence = deepcopy(sentence_lst)
            new_sentence[rep_idx] = word
            prob = self.ngram_model.predict_sentence_prob("".join(new_sentence))
            probs[word] = prob
        max_prob = sorted(probs.items(), key=lambda x: x[1], reverse=True)[0]
        return max_prob   # (word, prob)

    # 文本纠错
    def sentence_correct(self, sentence: str) -> str:
        init_prob = self.ngram_model.predict_sentence_prob(sentence)
        rep_list = []
        for i, ele in enumerate(sentence):
            cur_res = self.cal_replace_max_prob(i, sentence)
            if cur_res == -1:
                continue
            else:
                if cur_res[1] - init_prob >= self.threshold:
                    rep_list.append((i, cur_res[0]))
        new_sentence = list(sentence)
        for rep in rep_list:
            new_sentence[rep[0]] = rep[1]
        print("初始句子：", sentence)
        print("修正句子：", "".join(new_sentence))


# 加载同音词库
def load_tongyin_vocab(file_path):
    synoynms_dict = {}
    with open(file_path, encoding="utf-8") as file:
        for line in file.readlines():
            line = line.replace("\n", "")
            line = line.split(" ")
            synoynms_dict[line[0]] = list(line[1])
    return synoynms_dict


# 加载语料库
def load_corpus(file_path):
    train_corpus = []
    with open(file_path, encoding="utf-8") as file:
        for line in file.readlines():
            train_corpus.append(line.replace("\n", ""))
    return train_corpus


if __name__ == '__main__':
    syn_dict = load_tongyin_vocab("../datasets/tongyin.txt")
    corpus = load_corpus("../datasets/财经.txt")
    ngm = NGramModel(corpus, 3)
    sc = SentenceCorrect(syn_dict, ngm)
    sc.sentence_correct("每国货币政策空间不大")
