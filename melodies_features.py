import os
import pickle
import time
import numpy as np
from tqdm import tqdm
import torch
from itertools import chain

def explore_beat_changes(midi_file, start_time, end_time):
    beat_changes = 0
    for beat in midi_file.get_beats():  # Returns a list of beat locations, according to MIDI tempo changes
        if start_time <= beat <= end_time:
            beat_changes += 1
        elif beat > end_time:
            break
    return beat_changes


def explore_instruments(midi_file, start_time, end_time):
    sum_pitch, sum_velocity = 0, 0
    curr_word_has_drums = False
    num_of_instruments_in_same_time_of_word, num_of_notes = 0, 0
    for instrument in midi_file.instruments:
        note_in_range = False
        # explore all notes that this instrument was part of
        for note in instrument.notes:
            # when we should take note in account? remember we are now exploring for a specific word in a specific time
            # then this note is relevant for this word only if it was at the same time as the word
            take_note_into_account = start_time <= note.start and note.end <= end_time
            if take_note_into_account:
                curr_word_has_drums = True if instrument.is_drum else curr_word_has_drums
                note_in_range = True
                num_of_notes += 1
                sum_pitch += note.pitch
                sum_velocity += note.velocity
            else:  
                break
        if note_in_range:
            num_of_instruments_in_same_time_of_word += 1

    if num_of_notes == 0:
        avg_velocity, avg_pitch = 0, 0
    else:
        avg_velocity = sum_velocity / num_of_notes
        avg_pitch = sum_pitch / num_of_notes

    return avg_velocity, avg_pitch, num_of_instruments_in_same_time_of_word, num_of_notes, int(curr_word_has_drums)


def get_features_for_one_word(word_idx, time_per_word, midi_file):
    start_time = word_idx * time_per_word
    end_time = start_time + time_per_word
    # Features we want to extract:
    num_beat_changes = explore_beat_changes(midi_file, start_time, end_time)
    avg_velocity, avg_pitch, num_of_instruments, num_of_notes, has_drums = explore_instruments(midi_file,
                                                                                               start_time,
                                                                                               end_time)
    final_features = np.array([avg_velocity, avg_pitch, num_of_instruments, num_beat_changes, has_drums])
    return final_features


def try_load_melody(pkl_file_path):
    if os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            melody_train, melody_test = pickle.load(f)
        return [melody_train, melody_test]
    return pkl_file_path


def get_melodies_features_type_T1(melodies_list, sequence_length, all_songs_lyrics_words_as_int):
    features_for_all_songs = []
    for idx, midi_file in tqdm(enumerate(melodies_list), desc="Extracting features for melodies", unit="midi file",
                               total=len(melodies_list)):
        num_of_words_in_curr_song = len(all_songs_lyrics_words_as_int[idx])  # for example idx = 67,  len([15, 1444, 8484, 0, 3, 2, ...])
        midi_file.remove_invalid_notes()
        avg_time_per_word_in_curr_song = midi_file.get_end_time() / num_of_words_in_curr_song
        curr_song_features_by_word = []
        # step 1 get features for each word in curr song
        for word_idx_in_song in range(num_of_words_in_curr_song):  # iterate over every word and get the features for it
            features_for_curr_word = get_features_for_one_word(word_idx_in_song, avg_time_per_word_in_curr_song, midi_file)
            curr_song_features_by_word.append(features_for_curr_word)
        number_of_sequences = num_of_words_in_curr_song - sequence_length
        curr_song_features_sequences = []
        # step 2 divide to seq and get features
        for i in range(0, number_of_sequences, sequence_length):
            feature_seq = curr_song_features_by_word[i:i+sequence_length]
            curr_song_features_sequences.append(feature_seq)
        # step 3 add song-sequences-features to all songs
        features_for_all_songs.append(curr_song_features_sequences)
    return features_for_all_songs


# todo: shahar didnt touch this
def get_piano_roll_features(num_of_notes_per_word, piano_roll_matrix, word_idx):
    start_idx = word_idx * num_of_notes_per_word
    end_idx = start_idx + num_of_notes_per_word
    piano_roll_for_lyric = piano_roll_matrix[:, start_idx:end_idx].transpose()
    piano_roll_slice_sum = np.sum(piano_roll_for_lyric, axis=0)  # Sum each column into a single cell
    return piano_roll_slice_sum


# todo: shahar didnt touch this
def get_melodies_features_type_T2(melodies_list, sequence_length, all_songs_lyrics_words_as_int):
    features_for_all_songs = []
    for midi_idx, midi_file in tqdm(enumerate(melodies_list), desc="Extracting features for melodies", unit="midi file",
                               total=len(melodies_list)):
        num_of_words_in_curr_song = len(all_songs_lyrics_words_as_int[midi_idx])
        midi_file.remove_invalid_notes()
        """ "piano roll" refers to a matrix representation of MIDI data where each row corresponds to a specific note, 
        and each column corresponds to a specific time step or duration."""
        piano_roll_matrix = midi_file.get_piano_roll()
        num_of_notes_per_word = int(piano_roll_matrix.shape[1] / num_of_words_in_curr_song)  # Num of piano roll columns per word
        curr_features_for_curr_song = []
        for word_idx in range(num_of_words_in_curr_song):  # iterate over every word and get the features for it
            notes_features = get_piano_roll_features(num_of_notes_per_word, piano_roll_matrix, word_idx)
            # instrument_data = extract_features_for_one_word(word_idx, time_per_word, midi_file)
            # features = np.append(notes_features, instrument_data, axis=0)  # Concatenate them
            curr_features_for_curr_song.append(notes_features)

        number_of_sequences = num_of_words_in_curr_song - sequence_length
        features_seq_for_curr_song = []
        for i in range(0, number_of_sequences, sequence_length):
            sequence_features = curr_features_for_curr_song[i:i+sequence_length]
            features_seq_for_curr_song.append(sequence_features)

        features_for_all_songs.append(features_seq_for_curr_song)

    return features_for_all_songs


def normalize_melodies_features(features_in):  # features is 3 dim
    features = torch.tensor(np.array(list(chain(*features_in))))
    min_values = features.min(dim=1, keepdim=True)[0]
    max_values = features.max(dim=1, keepdim=True)[0]
    max_values = torch.where(max_values == min_values, max_values + 1e-7, max_values)
    normalized_features = (features - min_values) / (max_values - min_values)
    normalized_features = normalized_features.to(dtype=torch.float16)
    return normalized_features


def flatten_melodies_to_sequences(melodies):
    return np.concatenate([np.array([sequence], dtype=np.int32) for song in melodies for sequence in song])


def reconstruct_sequences_to_melodies(melodies, flatten_melodies):
    m = []
    i = 0
    for song in melodies:
        s = []
        for seq in song:
            seq = flatten_melodies[i]
            s.append(seq)
            i += 1
        m.append(s)
    return m


def create_melodies_data_sets(train_num, melodies_list, sequence_length, words_as_int_lyrics_matrix, pkl_file_path,
                              feature_extraction_type):
    print("Starting to extract melody features")
    melodies_extract_start_time = time.time()
    melody_or_path = try_load_melody(pkl_file_path)
    if isinstance(melody_or_path, list):
        melody_train = melody_or_path[0]
        melody_test = melody_or_path[1]
        return melody_train, melody_test
    else:
        pkl_file_path = melody_or_path
    """
    To understand what we did here one must understand the next params:
    "note" refers to a musical event representing the start and end of a specific pitch (גובה הצליל) being
    played on a particular MIDI channel. MIDI notes are fundamental to representing musical information
    in a digital format.
    
    "beats" typically refer to the rhythmic timing information that determines the tempo and timing of musical
    events within the composition.
    
    "pitch" In MIDI, refers to the frequency of a musical note, determining its perceived "highness" or "lowness."
     The pitch of a note is represented by a numerical value ranging from 0 to 127.
     
     "velocity" in MIDI, refers to the intensity or strength with which a note is played.
     It represents how forcefully a key is pressed on a MIDI controller or how strongly a note is triggered in
     a MIDI sequence. Velocity values range from 0 to 127, with higher values indicating louder or more forceful
     notes and lower values indicating softer or gentler notes.
    """
    print(f"You chose to use melodies feature extraction type: {feature_extraction_type}")

    if feature_extraction_type == 'T1':
        melody_features = get_melodies_features_type_T1(melodies_list, sequence_length, words_as_int_lyrics_matrix)
    elif feature_extraction_type == 'T2':
        melody_features = get_melodies_features_type_T2(melodies_list, sequence_length, words_as_int_lyrics_matrix)
    else:
        melody_features = []

    print(f"Melodies feature extraction type {feature_extraction_type} - Finished")

    # Normalization
    # step1 = flatten_melodies_to_sequences(melody_features)
    # step2 = normalize_melodies_features(step1)
    # step3 = reconstruct_sequences_to_melodies(melody_features, step2)
    # melody_features = step3 # normalize_melodies_features(melody_features)

    melody_train = melody_features[:train_num]
    melody_test = melody_features[train_num:]

    with open(pkl_file_path, 'wb') as f:
        content = [melody_train, melody_test]
        pickle.dump(content, f)
        print('Dumped melodies features files')

    melodies_extract_time_took = round((time.time() - melodies_extract_start_time) / 60, 2)
    print(f"Finished extracting melody features after {melodies_extract_time_took} minutes")
    return melody_train, melody_test


