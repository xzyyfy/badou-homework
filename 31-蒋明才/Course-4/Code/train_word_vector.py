"""
 #
 # @Author: jmc
 # @Date: 2023/3/11 16:59
 # @Version: v1.0
 # @Description: 使用Word2vec训练词向量
"""
from gensim.models import word2vec, Word2Vec


# CBOW + 层级 Softmax优化 的训练方式
def cbow_softmax_train(ds_path, model_save_path):
    sentences = word2vec.LineSentence(source=ds_path)
    model = Word2Vec(sentences, vector_size=256, window=5, min_count=3, sg=0, hs=1, epochs=1)
    model.save(model_save_path)
    return model


# CBOW + 负采样优化 的训练方式
def cbow_negative_train(ds_path, model_save_path):
    sentences = word2vec.LineSentence(source=ds_path)
    model = Word2Vec(sentences, vector_size=256, window=5, min_count=3, sg=0, hs=0, epochs=1)
    model.save(model_save_path)
    return model


# skip-gram + 层级 Softmax优化 的训练方式
def sg_softmax_train(ds_path, model_save_path):
    sentences = word2vec.LineSentence(source=ds_path)
    model = Word2Vec(sentences, vector_size=256, window=5, min_count=3, sg=1, hs=1, epochs=1)
    model.save(model_save_path)
    return model


# skip-gram + 负采样 的训练方式
def sg_negative_train(ds_path, model_save_path):
    sentences = word2vec.LineSentence(source=ds_path)
    model = Word2Vec(sentences, vector_size=256, window=5, min_count=3, sg=1, hs=0, epochs=1)
    model.save(model_save_path)
    return model


if __name__ == '__main__':
    model = cbow_softmax_train("../Datasets/word2vec_train.txt", "../Checkpoint/cbow_softmax.model")
    print("cbow_softmax", model.wv.similar_by_word("资金"))
    model = cbow_negative_train("../Datasets/word2vec_train.txt", "../Checkpoint/cbow_negative.model")
    print("cbow_negative", model.wv.similar_by_word("资金"))
    model = sg_softmax_train("../Datasets/word2vec_train.txt", "../Checkpoint/sg_softmax.model")
    print("sg_softmax", model.wv.similar_by_word("资金"))
    model = sg_negative_train("../Datasets/word2vec_train.txt", "../Checkpoint/sg_negative.model")
    print("sg_nagative", model.wv.similar_by_word("资金"))
