import numpy as np
import torch
import torch.nn as nn


class LSTMLyricsOnly(nn.Module):
    """
    Bidirectional LSTM model for lyrics only
    Architecture: input sequence --> embedding --> Bidirectional LSTM --> dropout layer --> output
    """

    def __init__(self, seed, vocabulary_size, embedded_vector_size, word2vec_matrix, num_LSTM_units, dropout_rate=0.0,
                 use_gpu=False):
        """
        :param vocabulary_size: number of words in the vocabulary
        :param embedded_vector_size: dimension of the word embedding
        """
        super(LSTMLyricsOnly, self).__init__()
        self.seed = seed
        self.setup_seeds()
        self.hidden_size = num_LSTM_units
        self.num_layers = 2

        self.device = torch.device("cuda") if (torch.cuda.is_available() and use_gpu) else torch.device("cpu")

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec_matrix, dtype=torch.int64), freeze=True,
                                                      padding_idx=0).to(self.device)
        self.lstm = nn.LSTM(input_size=embedded_vector_size, hidden_size=num_LSTM_units, bidirectional=True,
                            batch_first=True, num_layers=self.num_layers).to(self.device)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=num_LSTM_units * 2, out_features=vocabulary_size).to(self.device)

    def setup_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def forward(self, input_sequence):
        embedding_layer = self.embedding(input_sequence).float()
        LSTM_layer_hidden_output, state = self.lstm(embedding_layer)
        LSTM_dropout_layer = self.dropout(LSTM_layer_hidden_output)
        LSTM_last_unit_output = LSTM_dropout_layer[:, -1, :]
        fc_layer = self.fc(LSTM_last_unit_output)
        return fc_layer
