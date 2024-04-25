import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchtext.data import get_tokenizer


class InputSequencesCreator:
    def __init__(self, seed, embedding_matrix_shape: tuple, validation_size, sequence_len):
        self.seed = seed
        self.setup_seeds()
        self.num_words = embedding_matrix_shape[0]  # total number of words
        self.embedding_word_size = embedding_matrix_shape[1]  # == 300
        self.validation_size = validation_size
        self.sequence_len = sequence_len

    def setup_seeds(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def get_word2vec_matrix(self, model_dict, index2word: dict):
        """ for each integer that represents a specific word we insert the word embedding vector to the matrix at
        the index that correspond to the integer. """
        matrix_shape = (self.num_words, self.embedding_word_size)
        word_vectors_matrix = np.zeros(matrix_shape, dtype=int)
        for i, word in index2word.items():
            try:
                word_vectors_matrix[i] = model_dict[word]
            except Exception:
                continue
        return word_vectors_matrix

    def _convert_next_words_to_one_hot(self, sequences, next_encoded_words):
        next_words_one_hot = np.zeros(shape=(len(sequences), self.num_words), dtype=int)
        for i, encoded_word in enumerate(next_encoded_words):
            next_words_one_hot[i, encoded_word] = 1
        return next_words_one_hot

    def _create_sequences(self, words_as_int_lyrics_list, sequence_len, melodies_train=None):
        sequences_for_model_input = []
        next_encoded_words = []
        melodies_sequences_for_input = []
        melodies_song_idx = 0
        melodies_for_curr_song = []
        j = 0
        """ Loop all songs """
        for words_as_int_lyrics_song in words_as_int_lyrics_list:
            if melodies_train is not None:
                melodies_for_curr_song = melodies_train[melodies_song_idx]
                j = 0
            """ For every song loop number of sequences that should be in this song """
            for i in range(0, len(words_as_int_lyrics_song)-sequence_len, sequence_len):
                # lyrics
                curr_sequence = words_as_int_lyrics_song[i:i+sequence_len]
                sequences_for_model_input.append(curr_sequence)
                next_encoded_words.append(words_as_int_lyrics_song[i+sequence_len])
                # melodies
                if melodies_train is not None:
                    melody_curr_sequence_curr_song = melodies_for_curr_song[j]
                    melodies_sequences_for_input.append(melody_curr_sequence_curr_song)
                    j += 1
            melodies_song_idx += 1
        sequences_for_model_input = np.array(sequences_for_model_input)
        next_one_hot_words = self._convert_next_words_to_one_hot(sequences_for_model_input, next_encoded_words)
        return sequences_for_model_input, melodies_sequences_for_input, next_one_hot_words

    def _create_data_sets_dict(self, train_words_as_int_lyrics_list, test_words_as_int_lyrics_list, melodies_train=None):
        x_train_lyrics, x_train_melodies, y_train = self._create_sequences(train_words_as_int_lyrics_list,
                                                                           self.sequence_len,
                                                                           melodies_train)
        x_train_lyrics, x_val_lyrics, x_train_melodies, x_val_melodies = \
            self.separate_train_validation_sets(x_train_lyrics, x_train_melodies, self.validation_size)
        y_train, y_val, _, _ = self.separate_train_validation_sets(y_train,[], self.validation_size)
        x_test, _, y_test = self._create_sequences(test_words_as_int_lyrics_list, self.sequence_len)

        if melodies_train is not None:
            return {'train_set': (x_train_lyrics, x_train_melodies, y_train),
                    'validation_set': (x_val_lyrics, x_val_melodies, y_val),
                    'test_set': (x_test, y_test)}
        elif melodies_train is None:
            return {'train_set': (x_train_lyrics, y_train),
                    'validation_set': (x_val_lyrics, y_val),
                    'test_set': (x_test, y_test)}

    @staticmethod
    def separate_train_validation_sets(train_dataset, melodies_dataset, validation_size):
        list_of_indexes = np.arange(len(train_dataset))
        np.random.shuffle(list_of_indexes)
        train_validation_seperator = int(len(train_dataset) * validation_size)
        validation_indexes = list_of_indexes[:train_validation_seperator]
        train_indexes = list_of_indexes[train_validation_seperator:]

        lyrics_val_set = [train_dataset[idx] for idx in validation_indexes]
        lyrics_train_set = [train_dataset[idx] for idx in train_indexes]

        melodies_train_set, melodies_val_set = [], []
        if melodies_dataset:  # not empty list
            melodies_val_set = [melodies_dataset[idx] for idx in validation_indexes]
            melodies_train_set = [melodies_dataset[idx] for idx in train_indexes]

        return lyrics_train_set, lyrics_val_set, melodies_train_set, melodies_val_set

    @staticmethod
    def lyrics_to_sequences(lyrics, word2vec_dict):
        tokenizer = get_tokenizer('basic_english')
        vocabulary = set()
        for lyric in lyrics:
            tokens = tokenizer(lyric)
            for token in tokens:
                if token in word2vec_dict:
                    vocabulary.update([token])
        word_to_index = {word: index for index, word in enumerate(vocabulary)}  # Create word-to-index mapping
        words_as_int_lyrics = []  # Convert lyrics to sequences of integers
        for lyric in lyrics:
            tokens = tokenizer(lyric)
            curr_word_as_int_lyrics = [word_to_index[token] for token in tokens if token in word_to_index]
            words_as_int_lyrics.append(curr_word_as_int_lyrics)

        index_to_word = {index: word for word, index in word_to_index.items()}
        return words_as_int_lyrics, index_to_word, tokenizer

    def create_torch_data_loaders(self, train_size, words_as_int_lyrics_list, batch_size, melodies_train=None,
                                  model_name='LSTMLyricsOnly'):

        train_words_as_int_lyrics_list = words_as_int_lyrics_list[:train_size]
        test_words_as_int_lyrics_list = words_as_int_lyrics_list[train_size:]
        data_sets_dict = self._create_data_sets_dict(train_words_as_int_lyrics_list, test_words_as_int_lyrics_list, melodies_train)

        train_loader, validation_loader = None, None

        if model_name == 'LSTMLyricsOnly':
            train_input_lyrics = torch.tensor(np.array(data_sets_dict['train_set'][0])).long()
            train_y = torch.tensor(np.array(data_sets_dict['train_set'][1])).long()
            validation_input_lyrics = torch.tensor(np.array(data_sets_dict['validation_set'][0])).long()
            validation_y = torch.tensor(np.array(data_sets_dict['validation_set'][1])).long()

            train_tensor_dataset = TensorDataset(train_input_lyrics, train_y.long())
            train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
            validation_tensor_dataset = TensorDataset(validation_input_lyrics.long(), validation_y.long())
            validation_loader = DataLoader(validation_tensor_dataset, batch_size=batch_size, shuffle=True)

        elif model_name == 'LSTMLyricsMelodies':
            train_input_lyrics = torch.tensor(np.array(data_sets_dict['train_set'][0])).long()
            train_input_melodies = torch.tensor(np.array(data_sets_dict['train_set'][1])).long()
            train_y = torch.tensor(np.array(data_sets_dict['train_set'][2])).long()
            validation_input_lyrics = torch.tensor(np.array(data_sets_dict['validation_set'][0])).long()
            validation_input_melodies = torch.tensor(np.array(data_sets_dict['validation_set'][1])).long()
            validation_y = torch.tensor(np.array(data_sets_dict['validation_set'][2])).long()

            train_tensor_dataset = TensorDataset(train_input_lyrics, train_input_melodies, train_y.long())
            train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
            validation_tensor_dataset = TensorDataset(validation_input_lyrics.long(), validation_input_melodies.long(), validation_y.long())
            validation_loader = DataLoader(validation_tensor_dataset, batch_size=batch_size, shuffle=True)


        return train_loader, validation_loader
