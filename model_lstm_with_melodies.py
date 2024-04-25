import numpy as np
import torch
import torch.nn as nn

class LSTMLyricsMelodies(nn.Module):
    def __init__(self, seed, vocabulary_size, embedded_vector_size, word2vec_matrix,
                 num_LSTM_units, num_melodies_features, dropout_rate=0.0, use_gpu=False):
        super(LSTMLyricsMelodies, self).__init__()
        self.seed = seed
        self.setup_seeds()
        self.hidden_size = num_LSTM_units
        self.num_layers = 2

        self.device = torch.device("cuda") if (torch.cuda.is_available() and use_gpu) else torch.device("cpu")

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec_matrix, dtype=torch.int64), freeze=True,
                                                      padding_idx=0).to(self.device)
        self.lstm = nn.LSTM(input_size=embedded_vector_size+num_melodies_features, hidden_size=num_LSTM_units,
                            bidirectional=True, batch_first=True, num_layers=self.num_layers).to(self.device)
        self.dropout = nn.Dropout(p=dropout_rate)
        # times 2 because of bidirectional LSTM
        self.fc = nn.Linear(in_features=num_LSTM_units*2, out_features=vocabulary_size).to(self.device)
        # dim=1 to apply softmax along the dimension that represents the vocabulary
        self.softmax = nn.Softmax(dim=1).to(self.device)  # TODO: no softmax in model

    def setup_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def forward(self, lyrics_input_sequence, melodies_input_sequence):
        embedding_layer = self.embedding(lyrics_input_sequence).float()
        if embedding_layer.size()[0] == 1:  # when running test phase to generate_lyrics
            concatenate_features_layer = torch.cat((embedding_layer, melodies_input_sequence.unsqueeze(0).float()), dim=2)
        else:  # when running train phase
            concatenate_features_layer = torch.cat((embedding_layer, melodies_input_sequence), dim=2)
        LSTM_layer_hidden_output, state = self.lstm(concatenate_features_layer)
        LSTM_dropout_layer = self.dropout(LSTM_layer_hidden_output)
        LSTM_last_unit_output = LSTM_dropout_layer[:, -1, :]
        fc_layer = self.fc(LSTM_last_unit_output)
        return fc_layer
