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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from model4urlcls import FraudUrlDataset
import numpy as np

from pr import pr

criterion = nn.CrossEntropyLoss()
import matplotlib.pyplot as plt
softmax = nn.Softmax()

def test(model, test_loader, y):
    # 在测试集上测试模型
    test_loss = 0.0
    test_acc = 0.0
    model.load_state_dict(torch.load(model_dir))
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

    model_dir = "result/all/ckpt/new/lstm13cls_best.ckpt"
    result_dir = 'test.txt'
    model_name = "bigru2dcnn"  # "grucnn" "lstm" "bigru2dcnn" "convselfattn"

    # 加载数据集
    url_file = "data/all_test.csv"
    df = pd.read_csv(url_file)

    X_test = df['url'].tolist()
    y_test = df['label'].tolist()
    y = label_binarize(y_test,classes=list(range(num_class)))

    # 定义训练集和测试集的数据集对象和数据加载器
    test_dataset = FraudUrlDataset(X_test, y_test, max_len=200)
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
    test(model, test_loader, y)