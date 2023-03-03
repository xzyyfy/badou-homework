import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

规则：区分英文字母、中文汉字、日语假名
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, word_length, alphabet):
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(len(alphabet), vector_dim)
        self.pool = nn.AvgPool1d(word_length)
        self.classify = nn.Linear(vector_dim,3)
        self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y=None):
        x = self.embedding(x)
        x = self.pool(x.transpose(1,2)).squeeze()
        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

def build_alphabet():
    words_en = "abcdefghij"
    words_cn = "的一了是我不在人们有"
    words_jp = "あいうえおかきくけこ"
    words = words_en+words_cn+words_jp
    alphabet = {}
    for index, word in enumerate(words):
        alphabet[word] = index
    alphabet['unk'] = len(alphabet)
    return alphabet

def build_model(alphabet, vector_dim, word_length):
    model = TorchModel(vector_dim,word_length,alphabet)
    return model

def build_sample(alphabet, word_length):
    lang = np.random.randint(3)
    x = [random.choice(list(alphabet.keys())) for _ in range(word_length)]
    x = [alphabet.get(word, alphabet['unk']) for word in x]
    y = np.zeros(3)
    for i in x:
        if i <10:
            y[0] = 1
        elif i<20:
            y[1] = 1
        else:
            y[2] = 1
    return x, y

def build_dataset(sample_num, alphabet, word_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_num):
        x, y = build_sample(alphabet,word_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


def evaluate(model, alphabet, word_length):
    model.eval()
    x,y = build_dataset(200, alphabet,word_length)
    num_en = 0
    num_cn = 0
    num_jp = 0
    
    correct, wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred,y):
            y_p = y_p.numpy()
            y_t = y_t.numpy()
            catagory = np.zeros(3)
            for i in range(len(y_p)):
                if y_p[i] >= 0.5:
                    catagory[i] = 1
            if (catagory == y_t).all():
                correct += 1
            else:
                wrong += 1
            
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def predict(model_path, alphabet_path, input_strings):
    vector_dim = 20
    word_length = 3
    alphabet = json.load(open(alphabet_path, "r", encoding="utf8"))
    model = build_model(alphabet, vector_dim, word_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([alphabet[word] for word in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    result = result.numpy()
    for i, input_string in enumerate(input_strings):
        catagory = ""
        if result[i][0] >= 0.5:
            catagory += "字母 "
        if result[i][1] >= 0.5:
            catagory += "汉字 "
        if result[i][2] >= 0.5:
            catagory += "假名 "
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, catagory, str(result[i])))
        
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    vector_dim = 20
    word_length = 3
    learning_rate = 0.005

    alphabet = build_alphabet()

    model = build_model(alphabet,vector_dim,word_length)

    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):
            x, y = build_dataset(batch_size, alphabet, word_length)
            optim.zero_grad()
            loss = model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model,alphabet,word_length)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("alphabet.json", "w", encoding="utf8")
    writer.write(json.dumps(alphabet, ensure_ascii=False, indent=2))
    writer.close()
    return   


if __name__ == "__main__":
    main()
    test_strings = ["ad一", "eあく", "一けj", "う们是", "いけc", "有くあ"]
    predict("model.pth", "alphabet.json", test_strings)