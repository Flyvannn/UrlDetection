import torch
import torch.nn as nn

# 定义模型
class ConvSelfAttn(nn.Module):
    def __init__(self, num_chars, embedding_dim, hidden_dim, num_classes, kernel_size=5, num_heads=4):
        super(ConvSelfAttn, self).__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x, _ = self.self_attn(x, x, x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

class CharCNNLSTMAttn(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_classes, num_layers=1, kernel_sizes=[3, 4, 5], dropout=0.5):
        super(CharCNNLSTMAttn, self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim)
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.conv_layers.append(nn.Conv1d(embedding_dim, hidden_size, kernel_size))

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size * len(kernel_sizes), hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Attention layer
        # self.self_attn = nn.MultiheadAttention(hidden_size*2, num_heads=8)
        self.attn = nn.Linear(hidden_size * 2, 1)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # Input shape: (batch_size, seq_len , input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Convolutional layers
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(x)  # (batch_size, hidden_size, seq_len - kernel_size + 1)
            conv_output = F.relu(conv_output)
            conv_output = F.max_pool1d(conv_output, conv_output.size(2))  # (batch_size, hidden_size, 1)
            conv_outputs.append(conv_output.squeeze(2))
        x = torch.cat(conv_outputs, 1)  # (batch_size, hidden_size * len(kernel_sizes))

        # LSTM layer
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_size * len(kernel_sizes))
        lstm_outputs, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)

        # Attention layer
        attn_weights = self.attn(lstm_outputs)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(lstm_outputs.transpose(1, 2), attn_weights).squeeze(2)  # (batch_size, hidden_size * 2)

        # Dropout layer
        attn_applied = self.dropout(attn_applied)

        # Fully connected output layer
        output = self.fc(attn_applied)

        return output

class TCN(nn.Module):
    def __init__(self, input_size, embedding_dim, output_size, num_channels=[64,128,256,512], kernel_size=3, dropout=0.5):
        super(TCN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.conv_layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = embedding_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv_layers.append(conv_layer)
        self.network = nn.Sequential(*self.conv_layers)
        self.fc_layer = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        x = self.network(x)
        x = x.transpose(1, 2)  # Required to fit the Linear layer
        x = self.fc_layer(x[:, -1, :])
        return x

class CharGRUCNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(hidden_dim*2, out_channels=hidden_dim, kernel_size=3)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, urls):
        embedded = self.embedding(urls)
        embedded = self.dropout(embedded)

        output, hidden = self.gru(embedded)
        output = output.permute(0, 2, 1)
        output = self.conv(output)
        output = output.permute(0, 2, 1)
        output1 = torch.mean(output, dim=1)
        output2 = torch.max(output, dim=1)[0]
        output = torch.cat([output1, output2], dim=-1)

        return self.fc(self.dropout(output))


class BiGRU_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=3, dropout_prob=0.2,
                 kernel_sizes=[3, 4, 5], num_filters=100):
        super(BiGRU_CNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # CNN layers
        self.conv1 = nn.Conv2d(1, num_filters, (kernel_sizes[0], hidden_size * 2))
        self.conv2 = nn.Conv2d(1, num_filters, (kernel_sizes[1], hidden_size * 2))
        self.conv3 = nn.Conv2d(1, num_filters, (kernel_sizes[2], hidden_size * 2))

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, seq_len)
        x = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
        x, _ = self.gru(x)  # shape: (batch_size, seq_len, hidden_size*2)

        # Apply convolutions
        x = x.unsqueeze(1)  # shape: (batch_size, 1, seq_len, hidden_size*2)
        x1 = nn.functional.relu(self.conv1(x)).squeeze(3)  # shape: (batch_size, num_filters, seq_len-kernel_sizes[0]+1)
        x2 = nn.functional.relu(self.conv2(x)).squeeze(3)  # shape: (batch_size, num_filters, seq_len-kernel_sizes[1]+1)
        x3 = nn.functional.relu(self.conv3(x)).squeeze(3)  # shape: (batch_size, num_filters, seq_len-kernel_sizes[2]+1)

        # Max pooling over time
        x1 = nn.functional.max_pool1d(x1, x1.size(2)).squeeze(2)  # shape: (batch_size, num_filters)
        x2 = nn.functional.max_pool1d(x2, x2.size(2)).squeeze(2)  # shape: (batch_size, num_filters)
        x3 = nn.functional.max_pool1d(x3, x3.size(2)).squeeze(2)  # shape: (batch_size, num_filters)

        # Concatenate pooled features
        x = torch.cat((x1, x2, x3), dim=1)  # shape: (batch_size, len(kernel_sizes)*num_filters)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 定义模型
class CharLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(hidden_dim*2, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, urls):
        embedded = self.embedding(urls)
        embedded = self.dropout(embedded)

        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, url_lens.cpu(), batch_first=True, enforce_sorted=False)
        # packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(embedded)
        # hidden = hidden.transpose(0, 1)
        # packed_output, _ = self.attn(packed_output, packed_output, packed_output)
        # packed_output = packed_output.mean(1)

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(hidden)
        return self.fc(output)