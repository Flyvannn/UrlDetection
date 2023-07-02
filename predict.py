import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model import *


def pred(model, model_dir, test_loader):
    # 在测试集上测试模型
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    pred = []
    for inputs in tqdm(test_loader, desc="Test"):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        _, predictions = torch.max(outputs, 1)
        pred.extend(predictions.cpu().detach().numpy())

    return pred

class UrlDataset(Dataset):
    def __init__(self, urls, max_len):
        self.urls = urls
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, index):
        url = self.urls[index]
        if len(url) < self.max_len:
            url += ' ' * (self.max_len - len(url))
        elif len(url) > self.max_len:
            url = url[:self.max_len]
        url = [ord(c) for c in url]
        # return torch.tensor(url), torch.tensor(url_len), torch.tensor(label)
        return torch.tensor(url)

if __name__ == '__main__':
    # 定义超参数
    batch_size = 128
    hidden_size = 128
    embed_size = 200
    vocab_size = 128
    num_class = 13

    model_name = "lstm"  # "grucnn" "lstm" "bigru2dcnn"
    model_dir = "result/all/ckpt/new/lstm13cls_best.ckpt"

    # 加载数据集
    url_file = "data/test(unlabeled).csv"
    df = pd.read_csv(url_file, header=None)
    df.columns = ['url']
    urls = df.url.tolist()

    test_dataset = UrlDataset(urls, max_len=200)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型并将其移动到GPU（如果可用）
    if model_name == "lstm":
        model = CharLSTM(input_dim=vocab_size, emb_dim=embed_size, hidden_dim=hidden_size, output_dim=num_class)
    elif model_name == "grucnn":
        model = CharGRUCNN(input_dim=vocab_size, emb_dim=embed_size, hidden_dim=hidden_size, output_dim=num_class)
    elif model_name == "bigru2dcnn":
        model = BiGRU_CNN(vocab_size=vocab_size, embedding_dim=embed_size, hidden_size=hidden_size, num_classes=num_class)
    elif model_name == "convselfattn":
        model = ConvSelfAttn(num_chars=vocab_size, embedding_dim=embed_size, hidden_dim=hidden_size, num_classes=num_class)
    else:
        print("error model name")
        exit(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 测试
    preds = pred(model, model_dir, test_loader)
    print(preds)

    df.insert(loc=1, column='label', value=preds)

    df.to_csv("data/test_result.csv", header=None, index=False)