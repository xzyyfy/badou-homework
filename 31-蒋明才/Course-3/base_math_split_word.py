"""
 #
 # @Author: jmc
 # @Date: 2023/3/2 18:54
 # @Version: v1.0
 # @Description: 基于正向最大匹配的分词方法
                 基于反向最大匹配的分词方法
"""


# 加载词典 以及 词典的最长字符串的长度
def load_dict(dict_path):
    vocabs = {}  # 使用字典存储，可快速查询
    max_len = 0
    with open(dict_path, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            vocab = line.split(" ")[0]
            max_len = max(max_len, len(vocab))
            vocabs[vocab] = 1
    return vocabs, max_len


# 加载分词的语料库
def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path, encoding="utf-8") as file:
        for line in file.readlines():
            corpus.append(line.replace("\n", ""))
    return corpus


# 正向最大匹配  利用单指针实现
def forward_match(text, vocabs, window):
    words = []
    len_ = len(text)
    left = 0
    while left <= len_ - 1:
        for w in range(window, 0, -1):
            child = text[left: left + w]
            if child in vocabs or len(child) == 1:
                words.append(child)
                left += len(child)
                break
    return words


# 反向最大匹配  基于单指针
def backward_match(text, vocabs, window):
    words = []
    len_ = len(text)
    left = max(0, len_ - window)
    while len(text) != 0:
        child = text[left: left+window]
        if child in vocabs or len(child) == 1:
            words.append(child)
            text = text[:left]
            left = max(0, left - window)
        else:
            left += 1
    return list(reversed(words))


# 语料库分词
def corups_split_words(corpus, vocabs, window):
    for ele in corpus:
        forward = forward_match(ele, vocabs, window)
        backward = backward_match(ele, vocabs, window)
        print(f"原文：{ele}\n 正向最大分词结果：{forward}\n 反向最大分词结果：{backward}")


if __name__ == '__main__':
    vocabs, max_len = load_dict("./datasets/dict.txt")
    corpus = load_corpus("./datasets/corpus.txt")
    corups_split_words(corpus, vocabs, max_len)
    # print(forward_match("主力合约突破21000元/吨重要关口", vocabs, max_len))
    # print(backward_match("主力合约突破21000元/吨重要关口", vocabs, max_len))