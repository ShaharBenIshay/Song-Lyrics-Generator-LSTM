import csv
import os
import pickle
import string
import pretty_midi
from tqdm import tqdm


class MidiDataLoader:

    def __init__(self, midi_dir):
        self.midi_dir = midi_dir
        self.mapper = self.build_lower_to_upper_mapper()
        self.punctuation = string.punctuation.split()  # מילות ניקוד
        self.encoding_type = 'utf-8'

    def build_lower_to_upper_mapper(self):
        lower_to_upper = {}
        for midi_file_name in os.listdir(self.midi_dir):
            if midi_file_name.endswith('.mid'):
                midi_file_name_lower = midi_file_name.lower()
                lower_to_upper[midi_file_name_lower] = midi_file_name
            else:
                print(f"file {midi_file_name} is not a midi file ")
        return lower_to_upper

    def extract_midi_files_for_melodies(self, midi_pkl_path, artists, songs_names):
        pretty_midi_songs = self._try_read_pkl(midi_pkl_path)
        if pretty_midi_songs is not None:  # success opening pkl file
            return pretty_midi_songs
        """ there is no pkl file -> create new ! """
        pretty_midi_songs = []
        num_songs = len(songs_names)
        for i in tqdm(range(num_songs)):
            artist, song_name = artists[i], songs_names[i]
            if song_name[0] == " ":
                song_name = song_name[1:]  # why? because some songs names are like:' name' instead of:'name'
            artist = artist.replace(" ", "_")
            song_name = song_name.replace(" ", "_")
            song_full_name = f"{artist}_-_{song_name}.mid"
            try:
                song_midi_name = self.mapper.get(song_full_name.lower())
                if song_midi_name is None:
                    print(f"Song name not found in mapper {song_midi_name} Compare to {song_full_name.lower()}")
                    continue
                midi_song_path = os.path.join(self.midi_dir, song_midi_name)
                pretty_midi_song = pretty_midi.PrettyMIDI(midi_song_path)
                pretty_midi_songs.append(pretty_midi_song)
            except Exception:
                print(f'probelm for {song_full_name} in get_midi_files')
                continue

        self._try_save_pkl(midi_pkl_path, pretty_midi_songs)
        return pretty_midi_songs

    def _try_save_pkl(self, midi_pkl_path, pretty_midi_songs):
        with open(midi_pkl_path, 'wb') as file:
            pickle.dump(pretty_midi_songs, file)

    def _try_read_pkl(self, midi_pkl_path):
        if os.path.exists(midi_pkl_path):
            with open(midi_pkl_path, 'rb') as file:
                pretty_midi_songs = pickle.load(file)
                return pretty_midi_songs
        return None

    def extract_midi_files_for_datasets(self, input_file, pkl_path, word2vec_dict):
        pkl_value = self._try_read_pkl(pkl_path)
        data = {'artists': [], 'songs_names': [], 'lyrics': []}
        if pkl_value is not None:
            data['artists'], data['songs_names'], data['lyrics'] = \
                pkl_value['artists'], pkl_value['songs_names'], pkl_value['lyrics']
            return data
        else:  # there is no pickle -> create new pickle file
            with open(input_file, newline='', encoding=self.encoding_type) as f:
                lines = csv.reader(f, delimiter=',', quotechar='|')
                for line in lines:
                    artist, song_name, song_lyrics = line[0], line[1], line[2]
                    if song_name[0] == " ":
                        song_name = song_name[1:]
                    song_file_name = f'{artist}_-_{song_name}.mid'.replace(" ", "_").lower()
                    if self._check_valid_midi_file(song_file_name):
                        song_lyrics = self._preprocess_lyrics(song_lyrics, word2vec_dict)
                        data['artists'].append(artist)
                        data['songs_names'].append(song_name)
                        data['lyrics'].append(song_lyrics)
                    else:
                        continue

        self._try_save_pkl(pkl_path, data)
        return data

    def _check_valid_midi_file(self, song_file_name):
        if song_file_name not in self.mapper:
            print(f"{song_file_name} not in mapper so we dont have midi data on it !")
            return False
        original_file_name = self.mapper.get(song_file_name)
        midi_file_path = os.path.join(self.midi_dir, original_file_name)
        try:
            pretty_midi.PrettyMIDI(midi_file_path)
            return True
        except Exception:
            print(f'Exception raised from mido using this file: {midi_file_path}')
            return False

    def _preprocess_lyrics(self, song_lyrics, word2vec_dict):
        tokens = self._lyrics_to_tokens(song_lyrics)
        tokens = self._preprocess_tokens(tokens)
        in_word2vec_func = lambda w: w.lower() in word2vec_dict
        tokens = [word.lower() for word in tokens if in_word2vec_func(word)]
        lyrics = ' '.join(tokens)
        return lyrics

    def _lyrics_to_tokens(self, lyrics):
        lyrics = lyrics.replace('&', '')
        lyrics = lyrics.replace('  ', ' ')
        lyrics = lyrics.replace('\'', '')
        lyrics = lyrics.replace('--', ' ')
        return lyrics.split()

    def _preprocess_tokens(self, pre_tokens):
        after_tokens = []
        for token in pre_tokens:
            for punc in self.punctuation:
                token = token.replace(punc, '')
            if token.isalpha():
                after_tokens.append(token)
        return after_tokens
