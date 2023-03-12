"""
 #
 # @Author: jmc
 # @Date: 2023/3/11 17:12
 # @Version: v1.0
 # @Description: 数据预处理
"""
import jieba
from tqdm import tqdm
jieba.initialize()


def split_word(ds_path, save_path):
    with open(save_path, "w", encoding="utf-8") as writer:
        with open(ds_path, encoding="utf-8") as file:
            for line in tqdm(file.readlines(), desc="构建Word2vec的训练集"):
                line = line.replace("\n", "")
                words = jieba.lcut(line)
                words = [ele for ele in words if ele != " "]
                writer.write(" ".join(words) + "\n")


if __name__ == '__main__':
    split_word("../Datasets/titles.txt", "../Datasets/word2vec_train.txt")
