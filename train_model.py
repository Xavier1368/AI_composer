import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  

# 設定資料路徑
MODEL_DIR = 'model'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 訓練參數
EPOCHS = 100
BATCH_SIZE = 8192

class MIDIDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class TransformerModel(nn.Module):
    def __init__(self, n_vocab, seq_length, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.positional_encoding = self._get_positional_encoding(seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_vocab)

    def _get_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, seq_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, 1) -> (batch, seq_len)
        x = x.squeeze(-1).long()
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = self.transformer(x)  # shape: (batch, seq_len, d_model)
        x = self.fc(x[:, -1, :])  # 使用最後一個時間步的輸出
        return x




def load_training_data():
    network_input = np.load(os.path.join(MODEL_DIR, 'network_input.npy'))
    network_output = np.load(os.path.join(MODEL_DIR, 'network_output.npy'))
    pitch_names = np.load(os.path.join(MODEL_DIR, 'pitch_names.npy'), allow_pickle=True)
    n_vocab = len(pitch_names)

    network_input = network_input / float(n_vocab)
    network_input = np.reshape(network_input, (network_input.shape[0], network_input.shape[1], 1))

    return network_input, network_output, n_vocab

def main():
    print("載入訓練資料...")
    network_input, network_output, n_vocab = load_training_data()

    dataset = MIDIDataset(network_input, network_output)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerModel(n_vocab=n_vocab, seq_length=network_input.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    save_path = os.path.join(MODEL_DIR, 'best_model_torch.pth')

    print("開始訓練...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0


        for inputs, targets in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        tqdm.write("Epoch [{}/{}] Loss: {:.4f}".format(epoch + 1, EPOCHS, epoch_loss))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            tqdm.write("保存最佳模型：{}".format(save_path))

    print("訓練完成！")

if __name__ == '__main__':
    main()
