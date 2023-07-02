import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model import *
from pr import pr
import numpy as np

softmax = nn.Softmax()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
writer = SummaryWriter("./tensorboard")
criterion = nn.CrossEntropyLoss()

# 设置随机种子
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 定义数据集类
class FraudUrlDataset(Dataset):
    def __init__(self, urls, labels, max_len):
        self.urls = urls
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, index):
        url = self.urls[index]
        label = self.labels[index]
        url_len = len(url)
        if len(url) < self.max_len:
            url += ' ' * (self.max_len - len(url))
        elif len(url) > self.max_len:
            url = url[:self.max_len]
        url = [ord(c) for c in url]
        # return torch.tensor(url), torch.tensor(url_len), torch.tensor(label)
        return torch.tensor(url), torch.tensor(label)

def train(model,train_loader,val_loader,num_epochs):
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step = 0
    min_loss = 10000
    # 训练模型
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            train_acc += torch.sum(predictions == labels.data)

            writer.add_scalar("train_step_loss", loss.item(), step)
            step += 1

            loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())

        # 计算平均损失和精度
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc.double() / len(train_loader.dataset)

        writer.add_scalar("train_loss", train_loss, epoch)
        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}'.format(
            epoch + 1, train_loss, train_acc))

        # 在验证集上测试模型
        val_loss = 0.0
        val_acc = 0.0
        model.eval()

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            val_acc += torch.sum(predictions == labels.data)

        # 计算平均损失和精度
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_acc.double() / len(val_loader.dataset)

        if val_loss < min_loss:
            torch.save(model.state_dict(), model_dir + model_name + str(num_class) + "cls_best.ckpt")
            min_loss = val_loss

        writer.add_scalar("valid_loss", val_loss, epoch)
        print('Val Loss: {:.6f} \tVal Accuracy: {:.6f}'.format(val_loss, val_acc))

    torch.save(model.state_dict(), model_dir + model_name + str(num_class) + "cls_last.ckpt")

def test(model, test_loader, y):
    # 在测试集上测试模型
    test_loss = 0.0
    test_acc = 0.0
    model.load_state_dict(torch.load(model_dir + model_name + str(num_class) + "cls_best.ckpt"))
    model.eval()

    pred = []
    labs = []
    probs = []

    for inputs, labels in tqdm(test_loader, desc="Test"):
        inputs = inputs.to(device)
        labs.extend(labels)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        y_prob = softmax(outputs).cpu().detach().numpy()
        probs.extend(y_prob)

        test_loss += loss.item() * inputs.size(0)
        _, predictions = torch.max(outputs, 1)
        pred.extend(predictions.cpu().detach().numpy())
        test_acc += torch.sum(predictions == labels.data)

    # 计算平均损失和精度
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc.double() / len(test_loader.dataset)
    cr = classification_report(labs, pred, digits=6)
    cm = confusion_matrix(labs, pred)

    print('Test Loss: {:.6f} \tTest Accuracy: {:.6f}'.format(test_loss, test_acc))
    print("classification report"+"\n", cr)
    print("confusion matrix"+"\n", cm)

    with open(result_dir, "w") as f:
        f.write(cr)
        f.write("\nconfusion matrix"+"\n")
        f.write(str(cm)+"\n")
        f.write("acc:{}".format(test_acc))

    pr(num_class, y, np.array(probs))

    return pred

if __name__ == '__main__':
    # 定义超参数
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001
    hidden_size = 128
    embed_size = 200
    vocab_size = 128
    num_class = 13

    data_name = "all"
    model_name = "lstm"  # "grucnn" "lstm" "bigru2dcnn"
    model_dir = f"result/pretrain/{data_name}/ckpt/new/"
    pre_train_model = "result/all/ckpt/new/lstm2cls_best.ckpt"
    result_dir = f"result/pretrain/{data_name}/{model_name}_{num_class}cls_pre_ft.txt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 加载数据集
    train_df = pd.read_csv(f"data/{data_name}_train.csv")
    test_df = pd.read_csv(f"data/{data_name}_test.csv")
    # df = pd.read_csv("data/urlData.csv")
    # urls = df['url'].tolist()
    # labels = df['label'].tolist()
    X_train = train_df['url'].tolist()
    X_test = test_df['url'].tolist()
    y_train = train_df['label'].tolist()
    y_test = test_df['label'].tolist()
    y = label_binarize(y_test, classes=list(range(num_class)))

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # 定义训练集和测试集的数据集对象和数据加载器
    train_dataset = FraudUrlDataset(X_train, y_train, max_len=200)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = FraudUrlDataset(X_test, y_test, max_len=200)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = FraudUrlDataset(X_val, y_val, max_len=200)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型并将其移动到GPU（如果可用）
    if model_name == "lstm":
        model = CharLSTM(input_dim=vocab_size, emb_dim=embed_size, hidden_dim=hidden_size, output_dim=2)
    elif model_name == "grucnn":
        model = CharGRUCNN(input_dim=vocab_size, emb_dim=embed_size, hidden_dim=hidden_size, output_dim=2)
    elif model_name == "bigru2dcnn":
        model = BiGRU_CNN(vocab_size=vocab_size, embedding_dim=embed_size, hidden_size=hidden_size, num_classes=2)
    elif model_name == "convselfattn":
        model = ConvSelfAttn(num_chars=vocab_size, embedding_dim=embed_size, hidden_dim=hidden_size, num_classes=2)
    else:
        print("error model name")
        exit(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(pre_train_model))  # 加载预训练模型
    if model_name == "bigru2dcnn":
        model.fc = nn.Linear(300, num_class)
    else:
        model.fc = nn.Linear(hidden_size * 2, num_class)
    model.to(device)

    # 训练
    train(model,train_loader,val_loader,num_epochs)

    # 测试
    test(model,test_loader,y)







