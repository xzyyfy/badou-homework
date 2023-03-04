import json


def produce_pre_dict(path):
    pre_dict = {}

    with open(path, encoding='utf8') as f:
        for line in f:
            word = line.split()[0]
            for i in range(1, len(word)):
                if word[:i] not in pre_dict:
                    pre_dict[word[:i]] = 0
                pre_dict[word] = 1
    return pre_dict


def cut_word(sentence, pre_dict):
    word_list = []
    start_index, end_index = 0, 1
    find_word = window = sentence[start_index: end_index]
    # 不在词典当中的词
    unk_word_list = []

    while start_index < len(sentence):
        if window not in pre_dict or end_index > len(sentence):
            # 如果从前缀直接过度到不在前缀字典中.如天罜教教义 主错拼为罜
            # 拷贝一份至未知词列表中,标明来源
            if find_word not in pre_dict:
                unk_word_list.append({sentence: find_word})
            word_list.append(find_word)
            start_index += len(find_word)
            end_index = start_index + 1
            find_word = window = sentence[start_index: end_index]
        elif 1 == pre_dict[window]:
            find_word = window
            end_index += 1
            window = sentence[start_index: end_index]
        elif 0 == pre_dict[window]:
            end_index += 1
            # 针对天罜教教义的情况,不能在这里直接给find_word赋值
            # 会无法分给未知词列表
            window = sentence[start_index: end_index]

    return word_list, unk_word_list


def main(cut_method, input_path, out_path, dict_path):
    pre_dict = produce_pre_dict(dict_path)
    writer = open(out_path, 'w', encoding='utf8')
    total_unk_word = []

    with open(input_path, 'r', encoding='utf8') as f:
        for line in f:
            word_list, unk_word_list = cut_method(line, pre_dict)
            writer.write('/'.join(word_list) + '\n')
            total_unk_word += total_unk_word
    writer.close()

    print('未发现未知词' if len(total_unk_word) == 0 else total_unk_word)


main(cut_word, './corpus.txt', 'output.txt', './dict.txt')

# string = '天罜教教义'
# current_pre_dict = produce_pre_dict('./dict.txt')
# _, total_unk_word = cut_word(string, current_pre_dict)
# print('未发现未知词' if len(total_unk_word) == 0 else '未知分词:', total_unk_word)
