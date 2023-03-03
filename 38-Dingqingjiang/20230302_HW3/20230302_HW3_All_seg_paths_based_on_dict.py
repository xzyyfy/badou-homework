#coding:utf8

'''
主要思路：
加载词典 ->
遍历语句中的每个字，并遍历以该字开始的所有长度语句，找可能词汇的结束位置 ->
建立每个字可能的词结束位点的字典，形成有向无环图 ->
遍历有向无环图，找所有可能路径 ->
将所有路径对应到字符
'''

#加载词典，返回字典和词典词最大长度, key: 词；value: 前缀
def load_word_dict(path):
    max_word_length = 0
    word_dict = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split(" ")[:1]
            word = ''.join(word) #word: list->string
            word_dict[word] = word[:1]
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length

#生成输入字符串所有路径的有向无环图，返回路径字典。字典：key：字符下标；value：该字符为前缀的词可能结束的位置下标
def get_DAG(sen_input, word_dict, max_word_len):
    DAG = {} # 有向无环图的字典
    N = len(sen_input) # 输入的长度
    for charno in range(N): # 遍历输入长度N
        tmplist = [] # 该位置字符的词可能结束位置
        l = charno # 指定查找词的结束位置
        while l < min(N, max_word_len) and sen_input[charno] in word_dict.values():
            if sen_input[charno:l] in word_dict:
                tmplist.append(l)
            l += 1
        if not tmplist:
            tmplist.append(charno+1)
        DAG[charno] = tmplist
    return DAG

# 找DAG中的所有路径,list;path<list>:路径起始节点
def find_all_path_from_DAG(DAG, path, paths = []):
    start = path[-1]
    if start in DAG:
        for val in DAG[start]:
            new_path = path + [val]
            paths = find_all_path_from_DAG(DAG, new_path, paths)
    else:
        paths += [path]
    return paths

# 路径下标转化为字符
def get_result(paths,sen_input):
    result = []
    for path in paths:
        l = 0
        tmplist = []
        while l+1 < len(path):
            tmplist.append(sen_input[path[l]:path[l+1]])
            l += 1
        result.append(tmplist)
    return result

def main():
    vocab, max_word_len = load_word_dict('E:\\AIWork\\badou-qinghua\\38-Dingqingjiang\\20230302_HW3\\dict.txt')
    sen_input = '欢迎新老师生前来就餐'
    DAG = get_DAG(sen_input, vocab, max_word_len)
    paths = find_all_path_from_DAG(DAG, [0], [])
    result = get_result(paths, sen_input)
    for i in result:
        print(" ".join(i))
    return

if __name__ == "__main__":
    main()








