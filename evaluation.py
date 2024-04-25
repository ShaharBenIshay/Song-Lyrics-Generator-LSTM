import random
import numpy as np
import pandas as pd
import torch
from torch.distributions import OneHotCategorical
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


class LyricsEvaluator:
    """Class to generate and evaluate lyrics"""

    def __init__(self, word2vec_dict, index_to_word, testing_root_lyrics_per_song=1, sequence_len=1):
        self.word2vec_dict = word2vec_dict
        self.testing_root_lyrics_per_song = testing_root_lyrics_per_song  # = 3
        self.sequence_len = sequence_len
        self.index_to_word = index_to_word

    def _find_common_words(self, lyrics):
        words_in_song = []  # add words that are also in the Word2Vec
        words_in_lyrics = lyrics.split()
        for word in words_in_lyrics:
            if word in self.word2vec_dict:
                words_in_song.append(word)
        return list(words_in_song)

    def _generate_lyrics_per_song(self, root_sequence_to_generate_from, model, model_name, required_length,
                                  curr_melodies):
        song_first_indices = []
        for word in root_sequence_to_generate_from:
            word_index = [k for k, v in self.index_to_word.items() if v == word][0]
            song_first_indices.append(word_index)
        root_words_input = np.asarray(song_first_indices).reshape((1, self.sequence_len))
        new_song_lyrics = []
        j = 0
        voc_prob = None
        if curr_melodies is not None:
            num_melody_features = len(curr_melodies[0])
        else:
            num_melody_features = 0
        for word_i in range(required_length):
            if model_name == 'LSTMLyricsOnly':  # Different input for lyrics alone and lyrics and melodies
                logits = model(torch.tensor(root_words_input))
                distribution = OneHotCategorical(logits=logits)
                voc_prob = distribution.probs
            elif model_name == 'LSTMLyricsMelodies':
                curr_melody_sequence_input = curr_melodies[j: j + self.sequence_len]
                diff = self.sequence_len - len(curr_melody_sequence_input)
                if diff > 0:
                    padding = ([0] * num_melody_features)
                    for n_insertion in range(diff):
                        curr_melody_sequence_input.append(padding)
                logits = model(torch.tensor(root_words_input), torch.tensor(np.array(curr_melody_sequence_input)))
                j += 1
                distribution = OneHotCategorical(logits=logits)
                voc_prob = distribution.probs
            word_index_array_size = voc_prob.size()[1]
            word_index_array = np.arange(word_index_array_size)  # Create an array of the word indices [0, ..., 7699]
            # This line select a word based on the predicted probabilities
            index_of_selected_word = random.choices(word_index_array, k=1, weights=voc_prob.detach().squeeze().numpy())
            selected_word = self.index_to_word.get(index_of_selected_word[0], 'UNK_NONE')
            index_of_selected_word_array = np.array(np.array(index_of_selected_word).reshape(1, 1))
            root_words_input = np.append(root_words_input, index_of_selected_word_array, axis=1)
            remove_index = 0
            root_words_input = np.delete(root_words_input, remove_index, 1)
            new_song_lyrics.append(selected_word)
        final_text = ' '.join(new_song_lyrics)
        return final_text

    def generate_lyrics(self, model_name, sequence_len, model, artists, lyrics, names, melodies) -> (list, list):
        all_songs_original_lyrics, all_songs_generated_lyrics = [], []
        melody_features_per_song_idx = 0
        for curr_artist, curr_name, curr_lyrics in zip(artists, names, lyrics):
            n_lyrics = curr_lyrics
            curr_lyrics = curr_lyrics.split()
            print('*' * 80)
            print(f'Original lyrics for the song {curr_name} by {curr_artist}: (Lyrics length={len(curr_lyrics)}) '
                  f'\n"{n_lyrics}"')
            required_len_to_generate = len(curr_lyrics) - sequence_len
            if melodies is not None:
                curr_song_melody = melodies[melody_features_per_song_idx]
                curr_song_melody = list(chain(*curr_song_melody))
            else:
                curr_song_melody = None
            curr_lyrics_in_word2vec = list(filter(lambda w: w in self.word2vec_dict, curr_lyrics))

            curr_sequence_to_generate_from = curr_lyrics_in_word2vec[0:sequence_len]
            curr_generated_text = self._generate_lyrics_per_song(curr_sequence_to_generate_from,
                                                                 model=model, model_name=model_name,
                                                                 required_length=required_len_to_generate,
                                                                 curr_melodies=curr_song_melody)
            curr_generated_list = curr_generated_text.split(' ')
            generated_text_final = curr_generated_list.copy()
            all_songs_generated_lyrics.append(generated_text_final)
            all_songs_original_lyrics.append(curr_lyrics_in_word2vec)
        return all_songs_original_lyrics, all_songs_generated_lyrics

    def cosine_similarity_for_generated_lyrics(self, true_lyrics, generated_lyrics, sequence_len, word2vec_dict):
        all_songs_scores = []
        num_songs = len(true_lyrics)
        for i in range(num_songs):
            curr_true_lyrics, curr_generated_lyrics = true_lyrics[i], generated_lyrics[i]
            curr_song_score = []
            for j in range(0, num_songs, sequence_len):
                """ compute cosine similarity between each corresponding sequence"""
                true_sequence = curr_true_lyrics[j: j + sequence_len]
                generated_sequence = curr_generated_lyrics[j: j + sequence_len]
                score = self._cosine_similarity_between_sequences(true_sequence, generated_sequence, word2vec_dict)
                curr_song_score.append(score)
            mean_score = np.mean(curr_song_score)
            all_songs_scores.append(mean_score)
        all_songs_mean_score = np.mean(all_songs_scores)
        return all_songs_mean_score  # return the mean cosine similarity between All sequences

    def _cosine_similarity_between_sequences(self, true_sequence, generated_sequence, word2vec_dict):
        true_sequence_mean_vector = np.mean([word2vec_dict[word] for word in true_sequence], axis=0).reshape(1, -1)
        generated_sequence_mean_vector = np.mean([word2vec_dict[word] for word in generated_sequence], axis=0).reshape(
            1, -1)
        score = cosine_similarity(true_sequence_mean_vector, generated_sequence_mean_vector)[0][0]
        return score

    def jaccard_similarity(self, true_lyrics, generated_lyrics):
        """ create corpus for the true lyrics and the generated lyrics """
        true_words = list(chain(*true_lyrics))
        set_true_words = set()
        for word in true_words:
            set_true_words.update([word])
        pred_words = list(chain(*generated_lyrics))
        set_pred_words = set()
        for word in pred_words:
            set_pred_words.update([word])
        """ compute Jaccard = intersection divided be the union"""
        intersection = len(set_true_words.intersection(set_pred_words))
        union = len(set_true_words.union(set_pred_words))

        if union == 0:
            return 0  # Avoid division by zero
        else:
            return intersection / union

    def compute_sentiment_analysis(self, true_lyrics, generated_lyrics):
        # generated => gen
        vaderAnalyzer = SentimentIntensityAnalyzer()
        results = {'True': {}, 'Generated': {}}
        avg_neg, avg_neu, avg_pos, avg_compound = \
            ('Average Negative', 'Average Neutral', 'Average Positive', 'Average Compound')
        true_sum_negative, true_sum_neutral, true_sum_positive, true_sum_compound = 0, 0, 0, 0
        gen_sum_negative, gen_sum_neutral, gen_sum_positive, gen_sum_compound = 0, 0, 0, 0
        n = len(true_lyrics)
        for true_song, generated_song in zip(true_lyrics, generated_lyrics):
            # for true:
            true_scores = vaderAnalyzer.polarity_scores(true_song)
            true_sum_negative += true_scores['neg']
            true_sum_neutral += true_scores['neu']
            true_sum_positive += true_scores['pos']
            true_sum_compound += true_scores['compound']
            # for generated:
            gen_scores = vaderAnalyzer.polarity_scores(generated_song)
            gen_sum_negative += gen_scores['neg']
            gen_sum_neutral += gen_scores['neu']
            gen_sum_positive += gen_scores['pos']
            gen_sum_compound += gen_scores['compound']
        # Compute AVG
        results['True'][avg_neg] = true_sum_negative / n
        results['True'][avg_neu] = true_sum_neutral / n
        results['True'][avg_pos] = true_sum_positive / n
        results['True'][avg_compound] = true_sum_compound / n
        results['Generated'][avg_neg] = true_sum_negative / n
        results['Generated'][avg_neu] = true_sum_neutral / n
        results['Generated'][avg_pos] = true_sum_positive / n
        results['Generated'][avg_compound] = true_sum_compound / n
        df = pd.DataFrame.from_dict(results, orient='index')
        df.reset_index(inplace=True)
        df.to_csv('vader_analysis.csv')

    def compute_polarity_subjectivity(self, true_lyrics, generated_lyrics):
        results = {'True': {}, 'Generated': {}}
        true_polarity, true_subjectivity = 0, 0
        gen_polarity, gen_subjectivity = 0, 0
        n = len(true_lyrics)
        """ for each song we calculate sentiment and ploarity"""
        for true_song, gen_song in zip(true_lyrics, generated_lyrics):
            # for true:
            true_blob = TextBlob(" ".join(true_song))
            true_sentiment = true_blob.sentiment
            true_polarity += true_sentiment.polarity
            true_subjectivity += true_sentiment.subjectivity
            # for generated:
            gen_blob = TextBlob(" ".join(gen_song))
            gen_sentiment = gen_blob.sentiment
            gen_polarity += gen_sentiment.polarity
            gen_subjectivity += gen_sentiment.subjectivity
        # compute AVG
        results['True']['Polarity'] = true_polarity / n
        results['True']['Subjectivity'] = true_subjectivity / n
        results['Generated']['Polarity'] = gen_polarity / n
        results['Generated']['Subjectivity'] = gen_subjectivity / n
        df = pd.DataFrame.from_dict(results, orient='index')
        df.reset_index(inplace=True)
        df.to_csv('blob_analysis.csv')

    def split_to_paragraphs(self, text, num_pargraphs):
        words = text
        pargraph_size = int(len(words) / num_pargraphs)
        song = []
        curr_paragraph = []
        for word in words:
            curr_paragraph.append(word)
            if len(curr_paragraph) == pargraph_size:
                song.append(curr_paragraph)
                curr_paragraph = []
        if curr_paragraph:
            song.append(" ".join(curr_paragraph).split())
        return song

    def paragraph_to_sentences(self, pargraph, num_sentences):
        sentences = []
        current_sentence = []
        sentence_size = int(len(pargraph) / num_sentences)
        for word in pargraph:
            current_sentence.append(word)
            if len(current_sentence) == sentence_size:
                sentences.append(" ".join(current_sentence))
                current_sentence = []
        if current_sentence:
            sentences.append(" ".join(current_sentence))
        return sentences

    def transform_lyrics_to_song(self, full_song, num_paragraphs=4, num_sentences=4):
        print(f"Now we will show our generated song:" + '\n')
        paragraphs = self.split_to_paragraphs(full_song, num_paragraphs)
        for paragraph in paragraphs:
            sentences = self.paragraph_to_sentences(paragraph, num_sentences)
            sentences.append('\n')
            for sentence in sentences:
                print(sentence)

    def show_generated_songs(self, generated_songs, num_paragraphs=4, num_sentences=4):
        for song in generated_songs:
            print('-' * 50)
            self.transform_lyrics_to_song(song, num_paragraphs, num_sentences)
            print('-' * 50)

    def cosine_similarity_between_songs(self, true_lyrics, generated_lyrics, word2vec_dict):
        all_songs_scores = []
        num_songs = len(true_lyrics)
        for i in range(num_songs):
            curr_true_lyrics, curr_generated_lyrics = true_lyrics[i], generated_lyrics[i]
            true_sequence_mean_vector = np.mean([word2vec_dict[word] for word in curr_true_lyrics], axis=0).reshape(1,
                                                                                                                    -1)
            generated_sequence_mean_vector = np.mean([word2vec_dict[word] for word in curr_generated_lyrics],
                                                     axis=0).reshape(1, -1)
            score = cosine_similarity(true_sequence_mean_vector, generated_sequence_mean_vector)[0][0]
            all_songs_scores.append(score)
        all_songs_mean_score = np.mean(all_songs_scores)
        return all_songs_mean_score
