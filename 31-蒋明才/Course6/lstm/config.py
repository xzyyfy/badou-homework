"""
 #
 # @Author: jmc
 # @Date: 2023/3/22 23:07
 # @Version: v1.0
 # @Description: 配置文件
"""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 64692
input_dim = 256
hidden_dim = 128
out_dim = 18
num_layers = 1
bidirectional = True
dropout = 0.1
pool = "mean"
batch_size = 32
epoch = 1000
lr = 1e-2
wait = 10
model_save_path = "./checkpoint/lstm.pt"
