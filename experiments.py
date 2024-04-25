import datetime
from torch.utils.tensorboard import SummaryWriter
from model_lstm_with_melodies import *
from evaluation import *
from word2vec_builder import *
from model_lstm_lyrics_only import *
from melodies_features import *
from midi_data_loader import *
from input_sequences_creator import *
from tqdm import tqdm
import itertools


def train_with_validation(model, loss_function, optimizer, train_loader, validation_loader, num_epochs,
                          device, tensor_board_signature):
    train_epoch_loss_value = 0.0
    train_epochs_progress_bar = tqdm(iterable=range(1, num_epochs + 1), total=num_epochs,
                                     desc="Training Epochs", unit="epoch", position=0, colour='green')
    writer = SummaryWriter(tensor_board_signature)
    batch_cnt = 1
    for epoch_idx in train_epochs_progress_bar:
        train_epoch_loss_value = 0.0
        model.train()
        train_batches_progress_bar = tqdm(iterable=enumerate(train_loader), total=len(train_loader),
                                          desc="Training Batches", unit="batch", position=1, colour='blue')
        for batch_idx, (inputs, targets) in train_batches_progress_bar:

            if device == torch.device("cuda"):
                inputs = inputs.to(device)
                targets = targets.to(device)

            # Forward pass
            model_outputs = model(inputs).to(device)
            train_loss_func = loss_function(model_outputs, targets.float())

            # Backward pass
            optimizer.zero_grad()
            train_loss_func.backward()

            optimizer.step()
            train_epoch_loss_value += train_loss_func.item()
            train_batches_progress_bar.set_postfix(train_batch_loss=train_loss_func.item())
            writer.add_scalar('Train_Loss/Batch', train_loss_func.item(), batch_cnt)
            batch_cnt += 1
        train_epoch_loss_value /= len(train_loader)
        train_epochs_progress_bar.set_postfix(train_epoch_loss=train_epoch_loss_value)
        # Validation step
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            validation_progress_bar = tqdm(iterable=enumerate(validation_loader), total=len(validation_loader),
                                           desc="Validation Batches", unit="batch", position=2)
            for batch_idx, (inputs, targets) in validation_progress_bar:

                if device == torch.device("cuda"):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                model_outputs = model(inputs).to(device)
                validation_loss += loss_function(model_outputs, targets.float()).item()
                validation_progress_bar.set_postfix(validation_loss=validation_loss)

            validation_loss /= len(validation_loader)
            writer.add_scalars('Loss_Epochs',
                               {"Train": train_epoch_loss_value, "Validation": validation_loss},
                               epoch_idx)
    writer.flush()
    writer.close()

    return train_epoch_loss_value


def train_with_validation_with_melodies(model, loss_function, optimizer, train_loader, validation_loader, num_epochs,
                                        device, tensor_board_signature):
    train_epoch_loss_value = 0.0
    train_epochs_progress_bar = tqdm(iterable=range(1, num_epochs + 1), total=num_epochs,
                                     desc="Training Epochs", unit="epoch", position=0, colour='green')
    writer = SummaryWriter(tensor_board_signature)
    batch_cnt = 1
    for epoch_idx in train_epochs_progress_bar:
        train_epoch_loss_value = 0.0
        model.train()
        train_batches_progress_bar = tqdm(iterable=enumerate(train_loader), total=len(train_loader),
                                          desc="Training Batches", unit="batch", position=1, colour='blue')
        for batch_idx, (inputs_lyrics, inputs_melodies, targets) in train_batches_progress_bar:

            if device == torch.device("cuda"):
                inputs_lyrics = inputs_lyrics.to(device)
                inputs_melodies = inputs_melodies.to(device)
                targets = targets.to(device)

            # Forward pass
            model_outputs = model(inputs_lyrics, inputs_melodies).to(device)
            train_loss_func = loss_function(model_outputs, targets.float())

            # Backward pass
            optimizer.zero_grad()
            train_loss_func.backward()
            optimizer.step()

            train_epoch_loss_value += train_loss_func.item()
            train_batches_progress_bar.set_postfix(train_batch_loss=train_loss_func.item())
            writer.add_scalar('Train_Loss/Batch', train_loss_func.item(), batch_cnt)
            batch_cnt += 1

        train_epoch_loss_value /= len(train_loader)
        train_epochs_progress_bar.set_postfix(train_epoch_loss=train_epoch_loss_value)

        # Validation step
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            validation_progress_bar = tqdm(iterable=enumerate(validation_loader), total=len(validation_loader),
                                           desc="Validation Batches", unit="batch", position=2)
            for batch_idx, (inputs_lyrics, inputs_melodies, targets) in validation_progress_bar:

                if device == torch.device("cuda"):
                    inputs_lyrics = inputs_lyrics.to(device)
                    inputs_melodies = inputs_melodies.to(device)
                    targets = targets.to(device)

                model_outputs = model(inputs_lyrics, inputs_melodies).to(device)
                validation_loss += loss_function(model_outputs, targets.float()).item()
                validation_progress_bar.set_postfix(validation_loss=validation_loss)

            validation_loss /= len(validation_loader)

            writer.add_scalars('Loss_Epochs',
                               {"Train": train_epoch_loss_value, "Validation": validation_loss},
                               epoch_idx)
    writer.flush()
    writer.close()
    return train_epoch_loss_value


def run_one_experiment(seed,
                       batch_size,
                       sequence_len,
                       feature_extraction_type,
                       model_name,
                       num_LSTM_units,
                       learning_rate,
                       num_epochs,
                       word2vec_dict,
                       dropout_rate,
                       validation_size=0.2):
    """ ----------------------------------------- Set Configurations ---------------------------------------------"""
    current_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lyrics_train_pkl = 'pkl_files/lyrics_train.pkl'
    lyrics_train_csv = 'data/lyrics_train_set.csv'
    lyrics_test_pkl = 'pkl_files/lyrics_test.pkl'
    lyrics_test_csv = 'data/lyrics_test_set.csv'
    midi_files_dir = 'midi_files/'
    midi_files_pkl = 'pkl_files/pretty_midi_songs.pkl'
    """ ----------------------------------------- Load Data & Preprocess -----------------------------------------"""
    embedding_size = word2vec_dict.vector_size
    midi_data_loader = MidiDataLoader(midi_files_dir)
    # Load data sets with midi files
    train_set_data = midi_data_loader.extract_midi_files_for_datasets(lyrics_train_csv, lyrics_train_pkl, word2vec_dict)
    test_set_data = midi_data_loader.extract_midi_files_for_datasets(lyrics_test_csv, lyrics_test_pkl, word2vec_dict)
    all_artists = train_set_data['artists'] + test_set_data['artists']
    all_song_names = train_set_data['songs_names'] + test_set_data['songs_names']
    all_lyrics = train_set_data['lyrics'] + test_set_data['lyrics']
    # Create sequences for data input
    words_as_int_lyrics_list, index_to_word, tokenizer = InputSequencesCreator.lyrics_to_sequences(all_lyrics,
                                                                                                   word2vec_dict)
    num_words_in_corpus = len(index_to_word)
    input_sequences_creator = InputSequencesCreator(seed, (num_words_in_corpus, embedding_size),
                                                    validation_size, sequence_len)
    word2vec_matrix = input_sequences_creator.get_word2vec_matrix(word2vec_dict, index_to_word)
    """ ----------------------------------------- Extract Melody Features ----------------------------------------"""
    train_size = len(train_set_data['lyrics'])
    if model_name == 'LSTMLyricsMelodies':
        melodies = midi_data_loader.extract_midi_files_for_melodies(midi_files_pkl, all_artists, all_song_names)
        melody_train, melody_test = create_melodies_data_sets(
            train_num=train_size,
            melodies_list=melodies,
            sequence_length=sequence_len,
            words_as_int_lyrics_matrix=words_as_int_lyrics_list,
            pkl_file_path=f'pkl_files/melodies_features_seqLen{sequence_len}_{feature_extraction_type}.pkl',
            feature_extraction_type=feature_extraction_type)
        num_melodies_features = len(melody_train[0][0][0])
    else:
        melodies, melody_train, melody_test, num_melodies_features = None, None, None, 0
    """ ----------------------------------------- Create Data Loaders --------------------------------------------"""
    train_loader, validation_loader = input_sequences_creator.create_torch_data_loaders(train_size,
                                                                                        words_as_int_lyrics_list,
                                                                                        batch_size=batch_size,
                                                                                        melodies_train=melody_train,
                                                                                        model_name=model_name)
    """ ------------------------------------------------ Build Model ---------------------------------------------"""
    model = None
    if model_name == 'LSTMLyricsOnly':
        model = LSTMLyricsOnly(seed=seed,
                               vocabulary_size=word2vec_matrix.shape[0],
                               embedded_vector_size=word2vec_matrix.shape[1],
                               word2vec_matrix=word2vec_matrix,
                               num_LSTM_units=num_LSTM_units,
                               use_gpu=True,
                               dropout_rate=dropout_rate)
    elif model_name == 'LSTMLyricsMelodies':
        model = LSTMLyricsMelodies(seed=seed,
                                   vocabulary_size=word2vec_matrix.shape[0],
                                   embedded_vector_size=word2vec_matrix.shape[1],
                                   word2vec_matrix=word2vec_matrix,
                                   num_LSTM_units=num_LSTM_units,
                                   num_melodies_features=num_melodies_features,
                                   use_gpu=False,
                                   dropout_rate=dropout_rate)
    """ --------------------------------------------- Train Model ------------------------------------------------"""
    if current_device == torch.device("cuda"):
        model.to(current_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = None
    tensor_board_signature = f'logs/{model_name}_{feature_extraction_type}'
    if model_name == 'LSTMLyricsOnly':
        train_loss = train_with_validation(model, loss_function, optimizer, train_loader, validation_loader,
                                           num_epochs=num_epochs, device=current_device,
                                           tensor_board_signature=tensor_board_signature)
    elif model_name == 'LSTMLyricsMelodies':
        train_loss = train_with_validation_with_melodies(model, loss_function, optimizer, train_loader,
                                                         validation_loader,
                                                         num_epochs=num_epochs, device=current_device,
                                                         tensor_board_signature=tensor_board_signature)

    """ --------------------------------------------- Generate Lyrics --------------------------------------------"""
    test_artists = test_set_data['artists']
    test_lyrics = test_set_data['lyrics']
    test_songs_names = test_set_data['songs_names']
    evaluator = LyricsEvaluator(word2vec_dict=word2vec_dict, index_to_word=index_to_word, sequence_len=sequence_len)
    original_lyrics_of_all_songs, generated_lyrics_of_all_songs = evaluator.generate_lyrics(model_name=model_name,
                                                                                            sequence_len=sequence_len,
                                                                                            model=model.to(
                                                                                                torch.device("cpu")),
                                                                                            melodies=melody_test,
                                                                                            artists=test_artists,
                                                                                            names=test_songs_names,
                                                                                            lyrics=test_lyrics)
    """ --------------------------------------------- Evaluation Methods --------------------------------------"""
    cos_sim_score_comparing_sequences = evaluator.cosine_similarity_for_generated_lyrics(original_lyrics_of_all_songs,
                                                                               generated_lyrics_of_all_songs,
                                                                               sequence_len, word2vec_dict)
    cos_sim_score_comparing_songs = evaluator.cosine_similarity_between_songs(original_lyrics_of_all_songs,
                                                                              generated_lyrics_of_all_songs,
                                                                              word2vec_dict)

    jaccard_score = evaluator.jaccard_similarity(original_lyrics_of_all_songs, generated_lyrics_of_all_songs)

    evaluator.compute_sentiment_analysis(original_lyrics_of_all_songs, generated_lyrics_of_all_songs)

    evaluator.compute_polarity_subjectivity(original_lyrics_of_all_songs, generated_lyrics_of_all_songs)

    evaluator.show_generated_songs(generated_songs=generated_lyrics_of_all_songs)

    """ ---------------------------------- Clear Memory & Save Results -------------------------------------------"""
    model = None
    midi_data_loader = None
    input_sequences_creator = None
    train_loader = None
    validation_loader = None
    word2vec_dict = None
    return train_loss , cos_sim_score_comparing_sequences, jaccard_score, cos_sim_score_comparing_songs


def run_different_experiments(batch_size_lst,
                              sequence_len_lst,
                              feature_extraction_type_lst,
                              model_name_lst,
                              num_LSTM_units_lst,
                              learning_rate_lst,
                              num_epochs_lst,
                              dropout_rate_lst):
    word2vec_bin = 'data/GoogleNews-vectors-negative300.bin'
    word2vec_dict = load_google_word2vec(word2vec_bin)

    experiments_results = {'TrainLoss': {},
                           'CosineSimilarity_Sequences': {},
                           'JaccardScore': {},
                           'BatchSize': {},
                           'SequenceLength': {},
                           'FeatureType': {},
                           'NumLSTMUnits': {},
                           'LearningRate': {},
                           'NumEpochs': {},
                           'Dropout': {},
                           'CosineSimilarity_Songs': {}}
    for model_name in model_name_lst:
        experiments_params_lst = list(itertools.product(batch_size_lst,
                                                        sequence_len_lst,
                                                        feature_extraction_type_lst,
                                                        num_LSTM_units_lst,
                                                        learning_rate_lst,
                                                        num_epochs_lst,
                                                        dropout_rate_lst))
        print(f'Starting experiments for model {model_name}:  ({len(experiments_params_lst)} experiments to run)')
        model_experiment_idx = 1
        timestamp_experiments_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        for params_lst in experiments_params_lst:
            print(f'Model {model_name} -- Experiment #{model_experiment_idx}')
            batch_size, sequence_len, feature_extraction_type, num_LSTM_units, learning_rate, num_epochs, dropout = params_lst

            if model_name == 'LSTMLyricsOnly' and feature_extraction_type == 'T2':
                continue

            curr_train_loss, curr_cos_sim_score, curr_jaccard_score, curr_cos_sim_score_comparing_songs = (
                run_one_experiment(seed=316399773,
                                   batch_size=batch_size,
                                   sequence_len=sequence_len,
                                   feature_extraction_type=feature_extraction_type,
                                   model_name=model_name,
                                   num_LSTM_units=num_LSTM_units,
                                   learning_rate=learning_rate,
                                   num_epochs=num_epochs,
                                   word2vec_dict=word2vec_dict,
                                   dropout_rate=dropout))

            experiment_name = f'Model-{model_name}-Experiment-{model_experiment_idx}'
            experiments_results['TrainLoss'][experiment_name] = round(curr_train_loss, 4)
            experiments_results['CosineSimilarity_Sequences'][experiment_name] = round(curr_cos_sim_score, 4)
            experiments_results['JaccardScore'][experiment_name] = round(curr_jaccard_score, 4)
            experiments_results['BatchSize'][experiment_name] = batch_size
            experiments_results['SequenceLength'][experiment_name] = sequence_len
            experiments_results['FeatureType'][experiment_name] = feature_extraction_type
            experiments_results['NumLSTMUnits'][experiment_name] = num_LSTM_units
            experiments_results['LearningRate'][experiment_name] = learning_rate
            experiments_results['NumEpochs'][experiment_name] = num_epochs
            experiments_results['Dropout'][experiment_name] = dropout
            experiments_results['CosineSimilarity_Songs'][experiment_name] = curr_cos_sim_score_comparing_songs

            print(f'Model {model_name} Experiment {model_experiment_idx} Results: -train_loss={curr_train_loss},\n '
                  f'-cosine_similarity={curr_cos_sim_score}, -jaccard_score={curr_jaccard_score}')
            model_experiment_idx += 1

        print(f'Finished experiments for model {model_name}')
        print('~' * 120)
        experiments_results_df = pd.DataFrame.from_dict(experiments_results)
        experiments_results_df.reset_index(inplace=True)
        results_path = f'experiments_results_{timestamp_experiments_start}.csv'
        experiments_results_df.to_csv(results_path, index=False)


if __name__ == '__main__':
    batch_size_lst = [32]
    sequence_len_lst = [5]
    feature_extraction_type_lst = ['T1']
    model_name_lst = ['LSTMLyricsMelodies']
    num_LSTM_units_lst = [256]
    learning_rate_lst = [0.001]
    num_epochs_lst = [5]
    dropout_rate_lst = [0.4]

    run_different_experiments(batch_size_lst=batch_size_lst,
                              sequence_len_lst=sequence_len_lst,
                              feature_extraction_type_lst=feature_extraction_type_lst,
                              model_name_lst=model_name_lst,
                              num_LSTM_units_lst=num_LSTM_units_lst,
                              learning_rate_lst=learning_rate_lst,
                              num_epochs_lst=num_epochs_lst,
                              dropout_rate_lst=dropout_rate_lst)
