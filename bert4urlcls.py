import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./tensorboard")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FraudURLClassifier(nn.Module):
    def __init__(self, bert_model):
        super(FraudURLClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_cls)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        output = self.dropout(output)
        output = self.fc1(output)
        output = nn.ReLU()(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

# Preprocess data
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\bhttps?://\S*\b', '', text)
    text = re.sub(r'\bwww\.\S*\.\S*\b', '', text)
    text = re.sub(r'\b\w+\.(com|org|net|gov|edu)\b', '', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', text)
    text = re.sub(r'\b(\d{1,3}\.){3}\d{1,3}(:\d+)?\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define custom dataset
class FraudURLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        # self.data = df
        self.text = urls
        self.label = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        text = str(self.text[index])
        label = self.label[index]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                  truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0],
                'label': torch.tensor(label, dtype=torch.long)}

# Define training and validation functions
def train(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    loop = tqdm(train_loader, total=len(train_loader))
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input_ids.size(0)
        train_acc += accuracy_score(label.cpu().numpy(), np.argmax(output.cpu().detach().numpy(), axis=1))
        loop.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        loop.set_postfix(loss=loss.item())
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader)
    return train_loss, train_acc

def validate(model, criterion, val_loader):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, label)
            val_loss += loss.item() * input_ids.size(0)
            val_acc += accuracy_score(label.cpu().numpy(), np.argmax(output.cpu().detach().numpy(), axis=1))
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader)
    return val_loss, val_acc

if __name__ == '__main__':
    num_cls = 13
    n_epochs = 30
    batch_size = 64
    max_len = 64

    data_name = "all"  # all access
    model_name = "bert"
    result_dir = f"result/{data_name}/{model_name}_{num_cls}cls.txt"

    # 加载数据集
    train_df = pd.read_csv(f"data/{data_name}_train.csv")
    test_df = pd.read_csv(f"data/{data_name}_test.csv")
    # df = pd.read_csv("data/urlData.csv")
    if num_cls == 2:
        train_df['label'] = train_df['label'].map(lambda x: 1 if x != 0 else 0)  # 把标签映射为两类，做二分类
        test_df['label'] = test_df['label'].map(lambda x: 1 if x != 0 else 0)  # 把标签映射为两类，做二分类

    # urls = df['url'].tolist()
    # labels = df['label'].tolist()
    train_df['text'] = train_df['url'].apply(preprocess)
    test_df['text'] = test_df['url'].apply(preprocess)

    X_train = train_df['text'].tolist()
    X_test = test_df['text'].tolist()
    y_train = train_df['label'].tolist()
    y_test = test_df['label'].tolist()

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(urls, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Split dataset into training, validation, and testing sets
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    # Instantiate BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(r'D:\Fly\URLsClassification\bert_pretrain')  # bert-base-uncased
    bert_model = BertModel.from_pretrained(r'D:\Fly\URLsClassification\bert_pretrain')

    # Instantiate custom classifier
    classifier = FraudURLClassifier(bert_model).to(device)

    # Define dataset and data loader for training, validation, and testing
    train_dataset = FraudURLDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = FraudURLDataset(X_val, y_val, tokenizer, max_len)
    test_dataset = FraudURLDataset(X_test, y_test, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # unfreeze_layers = ['layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'bert.pooler', 'fc1', 'fc2']
    # for name, param in classifier.named_parameters():
    #     param.requires_grad = False
    #     for ele in unfreeze_layers:
    #         if ele in name:
    #             param.requires_grad = True
    #             break

    # Define optimizer and loss function
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, classifier.parameters()),  lr=2e-5)
    optimizer = optim.AdamW(classifier.parameters(),  lr=2e-5)

    total_steps = len(X_train) * n_epochs / batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()

    # Train and validate the model
    for epoch in range(n_epochs):
        print("training...")
        train_loss, train_acc = train(classifier, optimizer, criterion, train_loader)
        scheduler.step()

        print("validing...")
        val_loss, val_acc = validate(classifier, criterion, val_loader)
        print('Epoch:', epoch + 1, 'Train Loss:', train_loss, 'Train Accuracy:', train_acc, 'Val Loss:', val_loss,
              'Val Accuracy:', val_acc)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("valid_loss", val_loss, epoch)

    writer.close()

    # Test the model on the testing set
    classifier.eval()
    test_loss = 0.0
    test_acc = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = classifier(input_ids, attention_mask)
            loss = criterion(output, label)
            test_loss += loss.item() * input_ids.size(0)
            test_acc += accuracy_score(label.cpu().numpy(), np.argmax(output.cpu().detach().numpy(), axis=1))
            predictions.extend(np.argmax(output.cpu().detach().numpy(), axis=1))
            labels.extend(label.cpu().numpy())
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader)

    cr = classification_report(labels, predictions, digits=6)
    cm = confusion_matrix(labels, predictions)
    print('Test Loss:', test_loss, 'Test Accuracy:', test_acc)
    print("classification report"+"\n", cr)
    print("confusion matrix"+"\n", cm)

    with open(result_dir.format(num_cls), "w") as f:
        f.write(cr)
        f.write("\nconfusion matrix"+"\n")
        f.write(str(cm)+"\n")
        f.write("acc:{}".format(test_acc))