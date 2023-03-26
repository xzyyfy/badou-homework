"""
 #
 # @Author: jmc
 # @Date: 2023/3/22 23:08
 # @Version: v1.0
 # @Description: 模型训练的文件
"""
import json
import torch
import torch.nn as nn
import config
import base_model
import loader
from loader import CustomDataloader
from torch import optim
from sklearn.metrics import precision_score, f1_score
from transformers import BertTokenizerFast
from loguru import logger
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm


class TextCNNTrain:
    def __init__(self, model, train_loader, dev_loader, lr, epoch, save_path, log_writer, wait=10):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epoch = epoch
        self.save_path = save_path
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.save_path = save_path
        self.log_writer = log_writer
        self.wait = wait
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.01)

    def dev(self, model, dev_loader):
        model.eval()
        with torch.no_grad():
            pred, target = [], []
            for _, (x, y) in enumerate(dev_loader):
                out = self.model.forward(x)
                pred += torch.argmax(out, dim=1).tolist()
                target += y.tolist()
            f1 = f1_score(target, pred, average="macro")
            acc = precision_score(target, pred, average="macro")
            return acc, f1

    def train(self):
        logger.info("开始训练......")
        best_f1, wait = 0, 0
        for epoch in range(1, self.epoch+1):
            self.model.train()
            tr_loss, tr_example, tr_step = 0, 0, 0
            for idx, (x, y) in enumerate(tqdm(self.train_loader, desc=f"第{epoch}轮训练中")):
                out = self.model.forward(x)
                loss = self.loss_func.forward(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item()
                tr_example += y.size(0)
                tr_step += 1
            tr_loss /= tr_step
            acc, f1 = self.dev(self.model, self.dev_loader)
            self.log_writer.add_scalar(f"lstm/train/loss", tr_loss, epoch)
            self.log_writer.add_scalar(f"lstm/dev/f1", f1, epoch)
            self.log_writer.add_scalar(f"lstm/dev/acc", acc, epoch)
            logger.info(f"Epoch: {epoch}/{self.epoch}, Loss: {tr_loss}, Dev_f1: {f1}, Dev_acc: {acc}")
            if f1 >= best_f1:
                wait = 0
                best_f1 = f1
                # 保存radius和类别中心
                torch.save(self.model.state_dict(), self.save_path)
                logger.info("save lstm model.")
            else:
                wait += 1
                if wait >= self.wait:
                    logger.info("model train early stop")
                    break


def main():
    writer = SummaryWriter(logdir="./log/" + datetime.datetime.now().strftime('%Y-%m-%d'))

    train_ds, vocabs = loader.load_dataset("../dataset/train_tag_news.json")
    dev_ds, _ = loader.load_dataset("../dataset/valid_tag_news.json")
    train_loader = CustomDataloader(train_ds, vocabs, device=config.device, batch_size=config.batch_size).gen_dataloader()
    dev_loader = CustomDataloader(dev_ds, vocabs, device=config.device, batch_size=config.batch_size).gen_dataloader()

    model = base_model.TextCNN(config.input_dim, config.out_dim, config.filter_num, config.filter_size, config.vocab_size)
    TextCNNTrain(model, train_loader, dev_loader, config.lr, config.epoch, config.model_save_path, writer, config.wait).train()


if __name__ == '__main__':
    main()
