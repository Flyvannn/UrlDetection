from thop import profile, clever_format
from torchsummary import summary
import torch
from model import *
from torchstat import stat

# 定义超参数
batch_size = 128
num_epochs = 100
learning_rate = 0.001
hidden_size = 128
embed_size = 200
vocab_size = 128
num_class = 13

model = CharLSTM(input_dim=vocab_size, emb_dim=embed_size, hidden_dim=hidden_size, output_dim=num_class)
# model.load_state_dict(torch.load('result/all/ckpt/new/lstm13cls_best.ckpt'))

# stat(model, (1,200))

x = torch.randint(65, 128, (16, 200))
print(x)

flops, params = profile(model, inputs=(x,))
macs, params = clever_format([flops,params],"%.3f")

print(' FLOPs: ', macs)   # 一般来讲，FLOPs是macs的两倍
print('params: ', params)


# summary(model, input_size=(200,))
# x = torch.randn(200,  requires_grad=True)
# flops, params = profile(model, (x, ))
# print("flops:{} params:{}".format(flops,params))
# print(summary(model, x))