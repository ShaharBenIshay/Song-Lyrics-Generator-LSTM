# Song-Lyrics-Generator-LSTM
before explaining about our project, we should note that this is a work done in "Deep Learning" class and we where instructed to use PyTorch and not Keras. 

## Introduction:
This project explores the use of Recurrent Neural Networks (RNNs) for music generation, specifically focusing on generating song lyrics that align with a provided melody. We aimed to develop a model that, given a starting sequence of words and a melody, could predict the next lyrics of the song, in a coherent and meaningful way.

While evaluating the generated lyrics with some evaluation metrics (that will be explained later), we employed the Cross-Entropy loss function during training to optimize the model's performance. Pre-defined melody files and corresponding lyrics were provided for training and testing purposes. To monitor progress during training, a 20% split of the training data was designated as a validation set.

We opted for an LSTM network architecture due to its success in handling sequential data. LSTMs excel at remembering past information, critical in this task where each lyric builds upon the preceding words (and melody). The model takes a sequence of lyrics as input and predicts the next word in the sequence. The length of this input sequence significantly impacts the model's accuracy, with longer sequences offering better context for prediction. We experimented with different sequence lengths to determine their influence on model performance.

During training, the network is fed actual lyric sequences. Once trained, we can generate lyrics for an entire song by providing a seed sequence (a starting set of words). The model then predicts the next word, appends it to the sequence, and repeats the process, akin to a moving window, to generate the entire lyric.

## Dataset Anlysis & Preprocessing:
Our data preparation and preprocessing are divided into two parts: data in the CSV files (train, test) and the data in the midi files. We created two classes to help us handle the data. "MidiDataLoader" is responsible for reading and tokenizing the csv file and extracting the midi files. The second class is "InputSequencesCreator" which is responsible for creating sequences from the songs’ lyrics. 

**Train and Test Data (CSV):**
We received two CSV files, one for the train and one for the test. Each file has information about the artists, song names, and the lyrics of the songs. It is important to note that we removed songs whose corresponding midi file were corrupted files. 

<img width="283" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/7d48de6a-48a5-4d31-a57e-53c75a67f2fe">

NOTE: after preparing the data we saved it as a pkl file, and every time we tried to read those pkl files to save time. 

**Preprocessing steps in MidiDataLoader:**
1.	Read each line in the CSV file. For every line, we extracted the name of the artist, the name of the song, and the song lyrics. 
2.	Then we checked if the midi file that corresponds to this song is a valid midi file. 
3.	Next, we converted the lyrics from being a big string to a list of smaller strings and we removed signs like ‘&’ or ‘\’. 
4.	We took the "tokens" from step 3 and removed punctuation. 
Before saving the tokens, we checked if the current token is alphabetic, if it is then we added it to the tokens list.
5.	The final step for preprocessing the CSV files was to check if we have an embedding representation for this token in the word2vec dictionary.
6.	Next, we started working on the midi files. For each midi file, we tried to create a “pretty midi” object from the file, if this was successful then we added the object to a list.

Final Vocabulary Size: 6419.
  
**Preprocessing steps in InputSequencesCreator:**
To create the sequences for the models’ input we performed some steps. 
NOTE: a user can predefine the sequence length.
1.	We looped over every song and updated our vocabulary for every token we saw and made sure we had an embedding representation for this token in our word2vec. 
2.	After that we gave each token an index (integer) and converted the song from words to integers that represent those words. For example, "do you love me" can be mapped to [3, 789, 22, 25, 1]. 
3.	The next step was to create a word2vec matrix from the integers that represent the words. For example, if the word "me" is mapped to an integer value of 1, then we look up the word “me” in the word2vec dictionary and extract the embedding vector. Then we inserted this vector into the first row of the matrix. 
4.	The next step was to create the sequences. We used a shifting window method in the size of a sequence length and added the next word after the sequence to the true prediction list (Y). This true prediction was one hot encoded (vector of zeros in the size of the vocabulary except 1 in the true prediction).
5.	The last step was to separate the sequences to train (with validation) and test. 

 
**Midi Files:**
This directory has a .midi file corresponding to each song in the CSV data. These midi files represent musical data, they store instructions for electronic musical instruments, computers, and more. To work with those files, we used the "pretty midi" package and extracted from each "pretty midi" object some of the features. We read this object’s documentation and tried to figure out which features could be beneficial for our model. 
We divided our extraction method into two types (T1, T2):
•	Method #1 – Instruments: in this method, we extracted features that are related to the beats, instruments, and notes. "Note" refers to a musical event representing the start and end of a specific pitch (גובה הצליל) being played on a particular MIDI channel. MIDI notes are fundamental to representing musical information in a digital format. Now we will explain about the notes’ features we used:
a.	Velocity – refers to the intensity or strength with which a note is played (ranging from 0 to 127). 
b.	Pitch – frequency of a musical note (ranging from 0 to 127). 
c.	#Instruments – the number of instruments that were played during a specific word in the song. 
d.	#Notes –the number of notes during a specific word in the song. 
e.	Had drum – 1 or 0 if during the word there was a drum sound. 
•	Method #2 – Piano rolls: "piano roll" is a visual representation of musical notes over time. When working with MIDI files, piano rolls refer to a matrix representation of MIDI data where each row corresponds to a specific note and each column corresponds to a specific time step or duration. In this extraction method type, we first calculated the number of piano rolls per word. We then computed the summed intensities of notes over time for the words, from the data in the piano rolls’ matrix.

NOTE: It is important to mention that we tried to normalize the MIDI files features but it didn’t improve the results, and mainly the pkl files including the normalized features were too big in size for our personal computers, which couldn’t handle such large files (The PyCharm IDE crashed because of memory issues).

## Project Design:
Our project contains the next python files:
NOTE - we decide to implement some of your code as classes for modularity and convenient reasons.
word2vec_builder.py - This file is related to building and utilizing the word2vec model. It defines functions to process lyric data, including cleaning the text, tokenization, and stop word removal. Functions are included to load a pre-trained word2vec model from Google.
input_sequences_creator.py - Contains a class that deals with creating designed data loaders for training our model.
midi_data_loader.py - Contains a class to handle the midi files. It preproccess the midi files and extract features. 
model_lstm_lyrics_only.py - Contains a model (also a class) of an LSTM that generates lyrics without melodies.
melodies_features.py - 
model_lstm_with_melodies.py - Contains a model (also a class) of an LSTM that generates lyrics with melodies.
experiments.py
evaluation.py
  
## Model Architecture:
  ### Without melodies:
  ### With melodies:

## Training Process:

## Evaluation:
  ### Metrics:
  ### Combination tried:
  ### Results:
  ### TensorBoard:

## Conclusions:

