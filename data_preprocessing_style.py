# -*- coding: utf-8 -*-
import os
import numpy as np
import pretty_midi
import argparse

# 設定基礎路徑
MIDI_ROOT = 'midi_dataset'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCE_LENGTH = 50


def get_notes_from_midi(midi_file_path):
    """
    從單一 MIDI 檔案提取音符 (pitch)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(note.pitch)
        return notes
    except Exception as e:
        return []


def load_notes_from_folder(folder_path):
    """
    讀取特定資料夾內所有 MIDI 檔案
    """
    all_notes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            file_path = os.path.join(folder_path, filename)
            notes = get_notes_from_midi(file_path)
            if notes:  # 如果成功讀到音符才加入
                all_notes.extend(notes)
    return all_notes


def prepare_sequences(notes, sequence_length, pitch_names=None):
    """
    整理成訓練用 input/output pair
    """
    if pitch_names is None:
        pitch_names = sorted(set(notes))

    note_to_int = {note: number for number, note in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in seq_in])
        network_output.append(note_to_int[seq_out])

    network_input = np.array(network_input)
    network_output = np.array(network_output)

    return network_input, network_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', required=True, help='要處理的音樂風格 (資料夾名稱)')
    args = parser.parse_args()

    style_folder = args.style
    style_path = os.path.join(MIDI_ROOT, style_folder)

    print("讀取 {} MIDI 資料...".format(style_folder))
    notes = load_notes_from_folder(style_path)
    print("總共讀取到 {} 個音符".format(len(notes)))

    if len(notes) < SEQUENCE_LENGTH:
        print("錯誤：{} 音符太少，無法製作序列 (需要超過 {} 個音符)！".format(style_folder, SEQUENCE_LENGTH))
        return

    # 載入全體 pitch 名單（統一編碼）
    pitch_names = np.load(os.path.join(MODEL_DIR, 'pitch_names.npy'), allow_pickle=True)

    network_input, network_output = prepare_sequences(notes, SEQUENCE_LENGTH, pitch_names)

    # 儲存特定風格的資料
    np.save(os.path.join(MODEL_DIR, 'network_input_{}.npy'.format(style_folder)), network_input)
    np.save(os.path.join(MODEL_DIR, 'network_output_{}.npy'.format(style_folder)), network_output)

    print("{} 資料已儲存！".format(style_folder))


if __name__ == '__main__':
    main()
