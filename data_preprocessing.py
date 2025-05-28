# -*- coding: utf-8 -*-
import os
import numpy as np
import pretty_midi

# 設定 MIDI 資料夾與輸出位置
MIDI_FOLDER = 'midi_dataset'
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


def load_dataset(midi_folder):
    """
    從資料夾讀取所有 MIDI 檔案，並收集所有音符
    """
    all_notes = []
    for root, _, files in os.walk(midi_folder):
        for filename in files:
            if filename.endswith('.mid') or filename.endswith('.midi'):
                file_path = os.path.join(root, filename)
                notes = get_notes_from_midi(file_path)
                if notes:
                    all_notes.extend(notes)
    return all_notes


def prepare_sequences(notes, sequence_length):
    """
    將音符轉換成訓練用的 input/output pair
    """
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

    return network_input, network_output, pitch_names, note_to_int


def main():
    print("讀取 MIDI 資料...")
    notes = load_dataset(MIDI_FOLDER)
    print("總共讀取到 {} 個音符".format(len(notes)))

    if len(notes) < SEQUENCE_LENGTH:
        print("錯誤：資料量太小，無法製作序列！")
        return

    network_input, network_output, pitch_names, note_to_int = prepare_sequences(notes, SEQUENCE_LENGTH)

    np.save(os.path.join(MODEL_DIR, 'network_input.npy'), network_input)
    np.save(os.path.join(MODEL_DIR, 'network_output.npy'), network_output)
    np.save(os.path.join(MODEL_DIR, 'pitch_names.npy'), pitch_names)
    np.save(os.path.join(MODEL_DIR, 'note_to_int.npy'), note_to_int)

    print("資料處理完成，已儲存至 model/ 資料夾！")


if __name__ == '__main__':
    main()