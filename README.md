# README.md

# AI Composer — LSTM 自動作曲專案

一個簡單易懂的 AI 自動作曲工程，包含：
- 讀取 MIDI 資料
- 訓練 LSTM 模型
- 生成新的旋律
- 轉出為 MIDI 檔案
- 支援按風格對模型輕量度復調

---

## 資料組織

```
ai_composer/
├── midi_dataset/         # 放不同風格的 MIDI 訓練資料 (jazz, pop, classical, rock)
│   ├── jazz/
│   ├── pop/
│   ├── classical/
│   └── rock/
├── model/                # 儲存訓練好的模型與資料
├── output/               # 產生的 MIDI 檔案
├── data_preprocessing.py # 全資料前處理
├── data_preprocessing_style.py # 按風格分別處理
├── train_model_torch.py        # 執行基礎 LSTM 訓練
├── fine_tune_style_torch.py    # 對特定風格輕量度復調
├── generate_music_torch.py     # 產生新的旋律
├── requirements.txt      # 安裝套件列表
└── README.md             # 說明文件
```

## 安裝套件

建議先建立獨立環境：

```bash
conda create -n ai_composer python=3.10
conda activate ai_composer
```

然後安裝所需套件：

```bash
pip install -r requirements.txt
```

如果 numpy 存在版本關聯問題，可以用

```bash
pip install numpy==1.24.3
```


## 執行流程

### 1. 處理全資料（建立 pitch_names.npy）

```bash
python data_preprocessing.py
```

### 2. 處理特定風格資料

例如：
```bash
python data_preprocessing_style.py --style jazz
```

### 3. 訓練基礎模型

```bash
python train_model_torch.py
```

### 4. 對特定風格輕量度復調

例如：
```bash
python fine_tune_style_torch.py --style jazz
```

### 5. 產生特定風格新旋律

例如：
```bash
python generate_music_torch.py --style jazz
```

### 6. 查看 output/

用音樂播放程式打開檢視自動作曲結果！🎶

