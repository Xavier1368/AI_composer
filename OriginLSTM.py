class LSTMModel(nn.Module):
    def __init__(self, n_vocab, seq_length):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=512, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.fc1(x[:, -1, :])
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
