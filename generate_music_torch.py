import os
import numpy as np
import torch
import torch.nn as nn
import pretty_midi
import argparse
import random

# 設定資料夾
MODEL_DIR = 'model'
OUTPUT_DIR = 'output'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成參數
GENERATE_LENGTH = 500
CHORD_CHANGE_EVERY = 16

# 各風格設定
STYLE_SETTINGS = {
    'pop': {
        'chords': [
            {'name': 'C', 'notes': [60, 64, 67]},
            {'name': 'G', 'notes': [55, 59, 62]},
            {'name': 'Am', 'notes': [57, 60, 64]},
            {'name': 'F', 'notes': [53, 57, 60]}
        ],
        'pitch_range': (60, 84),
        'rest_probability': 0.1,
        'jump_probability': 0.2,
    },
    'rock': {
        'chords': [
            {'name': 'Am', 'notes': [57, 60, 64]},
            {'name': 'G', 'notes': [55, 59, 62]},
            {'name': 'F', 'notes': [53, 57, 60]},
            {'name': 'E', 'notes': [52, 56, 59]}
        ],
        'pitch_range': (48, 72),
        'rest_probability': 0.05,
        'jump_probability': 0.5,
    },
    'classical': {
        'chords': [
            {'name': 'C', 'notes': [60, 64, 67]},
            {'name': 'G', 'notes': [55, 59, 62]},
            {'name': 'Am', 'notes': [57, 60, 64]},
            {'name': 'Em', 'notes': [52, 55, 59]}
        ],
        'pitch_range': (48, 84),
        'rest_probability': 0.25,
        'jump_probability': 0.1,
    },
    'jazz': {
        'chords': [
            {'name': 'Cmaj7', 'notes': [60, 64, 67, 71]},
            {'name': 'Dm7', 'notes': [62, 65, 69, 72]},
            {'name': 'G7', 'notes': [55, 59, 62, 65]},
            {'name': 'Fmaj7', 'notes': [53, 57, 60, 64]}
        ],
        'pitch_range': (55, 79),
        'rest_probability': 0.15,
        'jump_probability': 0.3,
    }
}

# 打擊音設定
DRUM_NOTES = {
    'kick': 36,
    'snare': 38,
    'hihat_closed': 42
}

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

def load_seed_and_vocab(style=None):
    if style:
        network_input = np.load(os.path.join(MODEL_DIR, f'network_input_{style}.npy'))
    else:
        network_input = np.load(os.path.join(MODEL_DIR, 'network_input.npy'))
        
    pitch_names = np.load(os.path.join(MODEL_DIR, 'pitch_names.npy'), allow_pickle=True)
    n_vocab = len(pitch_names)
    note_to_int = {note: number for number, note in enumerate(pitch_names)}
    int_to_note = {number: note for number, note in enumerate(pitch_names)}

    network_input = network_input / float(n_vocab)
    network_input = np.reshape(network_input, (network_input.shape[0], network_input.shape[1], 1))

    return network_input, n_vocab, int_to_note

def correct_to_chord(pitch, chord_notes, pitch_range):
    # 先修到最近的和弦音
    candidates = sorted(chord_notes)
    closest = min(candidates, key=lambda x: abs(x - pitch))
    # 再限制在pitch範圍內
    closest = max(min(closest, pitch_range[1]), pitch_range[0])
    return closest

def generate_midi(prediction_output, chord_sequence, output_path, style):
    midi = pretty_midi.PrettyMIDI()

    # 隨機 BPM
    if style == 'jazz':
        bpm = random.randint(100, 180)
    elif style == 'classical':
        bpm = random.randint(60, 100)
    else:
        bpm = random.randint(80, 140)

    print(f"🎵 本次設定 Tempo: {bpm} BPM")

    NOTE_DURATION = 30.0 / bpm

    # 插入Tempo Change
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(4, 4, 0)  # 4/4拍，從0秒開始
    )
    midi._PrettyMIDI__initial_tempo = bpm
    
    if style == 'jazz':
        melody_program = pretty_midi.instrument_name_to_program('Trumpet')
        chord_program = pretty_midi.instrument_name_to_program('Tuba')
    elif style == 'rock':
        melody_program = pretty_midi.instrument_name_to_program('Electric Guitar (clean)')
        chord_program = pretty_midi.instrument_name_to_program('Orchestra Hit')
    elif style == 'pop':
        melody_program = pretty_midi.instrument_name_to_program('Rock Organ')
        chord_program = pretty_midi.instrument_name_to_program('Voice Oohs')
    else:  # classical
        melody_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        chord_program = pretty_midi.instrument_name_to_program('String Ensemble 1')


    melody_track = pretty_midi.Instrument(program=melody_program)        # Piano
    chord_track = pretty_midi.Instrument(program=chord_program)        # Strings
    drum_track = pretty_midi.Instrument(program=0, is_drum=True)  # Drum Kit


    start_time = 0
    drum_counter = 0

    for i, note_info in enumerate(prediction_output):
        current_chord = chord_sequence[(i // CHORD_CHANGE_EVERY) % len(chord_sequence)]

        # Melody
        if note_info == 'rest':
            start_time += NOTE_DURATION
        else:
            pitch = int(note_info)
            note = pretty_midi.Note(
                velocity=random.randint(80, 110),
                pitch=pitch,
                start=start_time,
                end=start_time + NOTE_DURATION
            )
            melody_track.notes.append(note)
            start_time += NOTE_DURATION

        # Chords
        if i % CHORD_CHANGE_EVERY == 0:
            chord_start = start_time
            chord_end = chord_start + (CHORD_CHANGE_EVERY * NOTE_DURATION)
            for p in current_chord['notes']:
                chord_note = pretty_midi.Note(
                    velocity=80,
                    pitch=p,
                    start=chord_start,
                    end=chord_end
                )
                chord_track.notes.append(chord_note)

        # Drums
        drum_step = drum_counter % 8
        if drum_step == 0:
            drum_track.notes.append(pretty_midi.Note(velocity=100, pitch=DRUM_NOTES['kick'], start=start_time, end=start_time + 0.1))
        if drum_step == 4:
            drum_track.notes.append(pretty_midi.Note(velocity=100, pitch=DRUM_NOTES['snare'], start=start_time, end=start_time + 0.1))
        drum_track.notes.append(pretty_midi.Note(velocity=70, pitch=DRUM_NOTES['hihat_closed'], start=start_time, end=start_time + 0.1))

        drum_counter += 1

    midi.instruments.append(melody_track)
    midi.instruments.append(chord_track)
    if style != 'classical':
        midi.instruments.append(drum_track)

    midi.write(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default=None, help='指定風格 (pop, rock, classical, jazz)')
    parser.add_argument('--temperature', type=float, default=1.0, help='生成自由度')
    args = parser.parse_args()

    if args.style is None:
        style = random.choice(list(STYLE_SETTINGS.keys()))
        print(f"🎲 沒指定風格，自動隨機選擇：{style.upper()} 風格！")
    else:
        style = args.style.lower()
    assert style in STYLE_SETTINGS, f"不支援的風格：{style}"

    settings = STYLE_SETTINGS[style]
    chord_sequence = settings['chords']
    pitch_range = settings['pitch_range']
    rest_probability = settings['rest_probability']
    jump_probability = settings['jump_probability']

    model_path = os.path.join(MODEL_DIR, f'best_model_{style}_torch.pth') if os.path.exists(os.path.join(MODEL_DIR, f'best_model_{style}_torch.pth')) else os.path.join(MODEL_DIR, 'best_model_torch.pth')

    network_input, n_vocab, int_to_note = load_seed_and_vocab(style if os.path.exists(model_path) else None)
    seed_index = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[seed_index]

    model = TransformerModel(n_vocab=n_vocab, seq_length=pattern.shape[0]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"✅ 成功載入模型 {model_path}")
    print(f"開始生成 {style.upper()} 風格 Melody + Chords + Drums...")

    prediction_output = []
    pattern = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    for step in range(GENERATE_LENGTH):
        with torch.no_grad():
            prediction = model(pattern)

        prediction = prediction.view(-1)
        prediction = prediction / args.temperature
        probabilities = torch.softmax(prediction, dim=0)
        next_index = torch.multinomial(probabilities, 1).item()

        current_chord = chord_sequence[(step // CHORD_CHANGE_EVERY) % len(chord_sequence)]
        chord_notes = current_chord['notes']

        # 決定是否休止
        if random.random() < rest_probability:
            next_note = 'rest'
        else:
            next_note = int_to_note[next_index]
            # 加上跳躍限制
            if random.random() < jump_probability:
                # 大跳自由
                next_note = correct_to_chord(int(next_note), chord_notes, pitch_range)
            else:
                # 小跳靠近前一音
                prev_pitch = int(next_note) if prediction_output and prediction_output[-1] != 'rest' else 60
                next_note = prev_pitch + random.choice([-2, -1, 0, 1, 2])
                next_note = max(min(next_note, pitch_range[1]), pitch_range[0])

        prediction_output.append(next_note)

        new_input = next_index / float(n_vocab)
        new_input = torch.tensor([[[new_input]]], dtype=torch.float32).to(DEVICE)
        pattern = torch.cat((pattern[:, 1:, :], new_input), dim=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'generated_{style}.mid')
    generate_midi(prediction_output, chord_sequence, output_path, style)

    print(f"🎶 生成完成！存到：{output_path}")

if __name__ == '__main__':
    main()
