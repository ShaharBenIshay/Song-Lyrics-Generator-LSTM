import time
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import csv
import nltk
from nltk.corpus import stopwords
import string
import pickle
nltk.download('stopwords')
nltk.download('punkt')


def csv_to_list_of_lists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        list_of_lists = []
        for row in reader:
            lyrics_only = row[2]
            if row[3] != '':
                list_of_lists.append(row[3])
                list_of_lists.append(row[4])
                if row[5] != '':
                    list_of_lists.append(row[5])
                    list_of_lists.append(row[6])
            list_of_lists.append(lyrics_only)
            song_names = row[1]
            list_of_lists.append(song_names)
    step1 = list_of_lists
    return step1


def remove_punctuation(list_of_lists):
    # flattened_sentences = list(itertools.chain(*list_of_lists))
    flattened_sentences = list_of_lists
    table = str.maketrans('', '', string.punctuation)
    no_punctuation_sentences = [sentence.translate(table) for sentence in flattened_sentences]
    return no_punctuation_sentences


def tokenize(without_punctuation):
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in without_punctuation]
    return tokenized_sentences


def remove_stopwords(tokenized_data):
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [[word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_data]
    return filtered_sentences


def build_word2vec(path='lyrics_train_set.csv'):
    list_of_lists = csv_to_list_of_lists(path)
    without_punctuation = remove_punctuation(list_of_lists)
    tokenized_data = tokenize(without_punctuation)
    final_data = remove_stopwords(tokenized_data)
    model = Word2Vec(final_data, min_count=1, vector_size=300, workers=4)
    with open("pkl_files/word2vec_model_no_artists_count_1.pkl", 'wb') as f:
        pickle.dump(model, f)


def load_word2vec(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_google_word2vec(file_path):
    print("Loading Google's Word2Vec model...")
    start_time = time.time()
    word2vec_dict = KeyedVectors.load_word2vec_format(file_path, binary=True, datatype=np.float16)
    load_time_took = round((time.time() - start_time) / 60, 2)
    print(f"Google's Word2Vec model loaded successfully! Time took: {load_time_took} minutes")
    return word2vec_dict
