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

**Method #1** – Instruments: in this method, we extracted features that are related to the beats, instruments, and notes. "Note" refers to a musical event representing the start and end of a specific pitch (גובה הצליל) being played on a particular MIDI channel.
MIDI notes are fundamental to representing musical information in a digital format.
Now we will explain about the notes’ features we used:
* Velocity – refers to the intensity or strength with which a note is played (ranging from 0 to 127).
* Pitch – frequency of a musical note (ranging from 0 to 127).
* Number of Instruments – the number of instruments that were played during a specific word in the song.
* Number of Notes –the number of notes during a specific word in the song.
* Had drum – 1 or 0 if during the word there was a drum sound.

**Method #2** – Piano rolls: "piano roll" is a visual representation of musical notes over time. When working with MIDI files, piano rolls refer to a matrix representation of MIDI data where each row corresponds to a specific note and each column corresponds to a specific time step or duration. In this extraction method type, we first calculated the number of piano rolls per word. We then computed the summed intensities of notes over time for the words, from the data in the piano rolls’ matrix.

NOTE: It is important to mention that we tried to normalize the MIDI files features but it didn’t improve the results, and mainly the pkl files including the normalized features were too big in size for our personal computers, which couldn’t handle such large files (The PyCharm IDE crashed because of memory issues).

## Project Design:
Our project contains the next python files:
NOTE - we decide to implement some of your code as classes for modularity and convenient reasons.
* word2vec_builder.py - This file is related to building and utilizing the word2vec model. It defines functions to process lyric data, including cleaning the text, tokenization, and stop word removal. Functions are included to load a pre-trained word2vec model from Google.
* input_sequences_creator.py - Contains a class that deals with creating designed data loaders for training our model.
* midi_data_loader.py - Contains a class to handle the midi files and preproccess the songs lyrics. 
* model_lstm_lyrics_only.py - Contains a model (also a class) of an LSTM that generates lyrics without melodies.
* melodies_features.py - This file contains functions to extract features from the melodies in the midi files. 
* model_lstm_with_melodies.py - Contains a model (also a class) of an LSTM that generates lyrics with melodies.
* experiments.py - The Training loop with/without melodies and the experiments setup.
* evaluation.py - Since evluating process is a complex task we created a class called LyricsEvaluator to implement many kind of evluation metrics. 
  
## Model Architecture:
In our work, there are 2 main models, one with features from the midi files (melodies) and another without these features. Each model is built in its own class. Each class has a constructor and a forward method that we implemented. As a base for our models, we used a familiar RNN layer – LSTM. We used LSTM as our basis for two main reasons:
1. Capturing long-term dependencies: LSTMs are designed to capture those dependencies in sequential data like songs in this case. When generating lyrics to songs (or text in general), it's important to consider not only the immediately preceding words but also the context from farther back in the sequence (or ahead).
2. Handling variable-length sequences: it is not surprising that songs don’t have the same length, thus we should use a layer that can handle varying lengths.

To explore the model’s capacity to capture and represent information at different levels of abstraction, we used different numbers of units for the LSTM:
* 32 units – we used the smaller number of units of 32, to have a more compact representation of the data and potentially improve generalization by reducing the risk of overfitting. This can be beneficial when dealing with simpler patterns or datasets with limited complexity, therefore we can guess that a model without the melodies features will produce good results with this parameter value.
* 256 units – we also used a larger number of units to try to increase the model's capacity to capture complex patterns within the data. This can be beneficial for more complex datasets or tasks that require the model to learn more "soft" relationships between input features. We expect that with this parameter value, the models that include the melodies features will achieve better results.

### Without melodies:

**Model layers:**

Embedding layer – this layer receives the word2vec matrix (size = num of words in corpus X vector embedding size) as the layer's weights, these weights do not get updated (we don’t need to learned new word embedding).

LSTM layer – this layer is for learning from the sequences "through time". The input is the size of the embedding vector. After searching online, we decided to use 2 layers for this part (num_layers=2, should be sufficient). Another important aspect of this layer is that this is a Bidirectional RNN to enable the model to learn from different orders of the sequence (not only from start to end but also in the opposite direction). 

Dropout layer – we added this layer to the model because we wanted to explore if this additional layer can help (we experimented with varying rates). 

Fully-Connected layer – this layer input is twice the size of the LSTM units since the LSTM layer is bidirectional. The output of this layer is the size of our entire corpus. 

**Forward pass:**

As input, we get a sequence of "words as integers", for example- "i love you" sequence will be inputted to the model as [23, 44, 899]. 
First, we insert the input into the embedding layer, and then to the LSTM layer. From the LSTM layer, we get the hidden output. 
This output is then inputted to a dropout layer. 
After that we take the output of the last unit in the LSTM. 
The final step is to insert this output into the fully connected layer.

### With melodies:
-	Our second model differs from the first model mostly by the input sizes and number of inputs.
While the first model gets one input, the second model gets two inputs – the lyrics input and the melodies input. We can see this modification in the size of the LSTM layer.
This modification can also be seen in the forward pass, where we concatenate the input of the lyrics with the input of the melodies. 
-	Another important note is that we tried this model with two different kinds of features for the melodies (type T1 and type T2 as explained before). So, while the structure of the model remains quite the same, the size of the input features is different. 
Changes from Model #1:
-	Concatenate Features layer – this layer is only defined in the forward pass. Its purpose is to add to the embedding layer, which consists of the features for the lyrics, the features that were extracted for the melodies, either of type T1 or T2 (the type affects the input size).
-	In the forward pass of Model #2, after the embedding layer, the concatenation layer concatenates the melodies features to the embedding layer, and then the concatenation layer passes to the LSTM layer and so on, like in Model #1.

## Experiments: 
In this part, we focused on running multiple experiments to explore our models’ performance. We used different values for the following parameters in our experiments:

<img width="439" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/046e7fea-a542-45e2-b5a4-8b97bba9384b">

NOTES: 
We ran only 5 epochs for each experiment, due to our limited computation resources (our personal computers).
We also faced challenges when running the model with a sequence length of 1 as instructed, therefore we tried also longer sequences of length 5. The reason for this problem is that when using sequences of length 1, "more" data is created to be inputted of the data.

We started with running the above experiments and evaluating them only with scores – cosine and Jaccard, and training loss. We will later dive into more complex evaluations after drawing our final conclusions from the experiments.

<img width="581" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/b3b51a97-d6af-4edb-97d6-fe48313be6af">

-	First, the most significant conclusion is that most of the best experiments ran with a batch size of 32 and a sequence length of 5. We will fix those parameter values before continuing to experiment.
-	It is hard to determine what learning rate, melodies features type and dropout rate are the best.
-	Although only one experiment with num_LSTM_units=256 showed good results, still it is the 2nd best.

<img width="598" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/aea38cae-99c9-46d0-a050-e58f7f75f48f">

NOTE: For those experiments, we added another evaluation method – cosine similarity for songs.

We can see that now 256 units for LSTM achieve the best results. We will continue to the evaluation part with this parameter value.
We still got mixed results regarding the dropout rate. We decided to fix the dropout rate to 0.4 to use it as a method for regularization and generalization for the models.
We will further evaluate these 3 models in the 'Evaluation' section, after implementing the above conclusions.

## Evaluation
First, we will explain the various evaluation metrics we used to evaluate the lyrics generated:

* Cosine Similarity - Sequences: with this evaluation metric we computed the score for each sequence (according to the true song sequence) and then computed the average score among all sequences for all songs. Our comparison was made sequence-to-sequence because we care not only about predicting the "right" word but the “right” word in the "right" time. For example, if the word "love" was predicted in the first part, but the word "love" is at the end of the song – we don’t want to consider this as a perfect match.
  
* Cosine Similarity - Songs: we used this evaluation metric the same way as cosine similarity for sequences. The difference is that this metric is computed over whole songs, therefore it gives us a “feel” of how the entire generated song is similar to the original lyrics.

* Jaccard Similarity: with this metric, we wanted to evaluate if our model was able to predict the exact words in the original lyrics. 

* Sentiment Analysis: this metric is important since the task of predicting the exact same words is a very complicated task, but capturing the context and the sentiment of the song is a much more possible task. After searching online, we found VADER (Valence Aware Dictionary and sEntiment Reasoner), which is a lexicon and rule-based sentiment analysis tool used to calculate the polarity scores of the given text. VADER computes (the sentiment) for every song the neg (negative), neu(neutral), pos (positive), and compounds the scores.

* Polarity & Subjectivity: since we didn’t get much information from VADER, we used TextBlob. We used this object to compute the average polarity and subjectivity for each song – true and generated. Polarity indicates whether the text expresses a positive, negative, or neutral sentiment as explained above. Subjectivity is another NLP aspect, which refers to how the text expresses opinions, beliefs, and feelings. A score close to 0 indicates high objectivity, meaning that the text is predominantly factual and lacks subjective expressions. A score close to 1 indicates high subjectivity, meaning that the text contains a significant number of subjective opinions or emotions.

Now we will present the evaluations on the 3 experiments from the 'Experiments' section.
We ran those experiments with same parameters for the models:
-	Batch size = 32
-	Sequence Length = 5
-	Num LSTM Units = 256
-	Learning Rate = 0.001
-	Number of Epochs = 5
-	Dropout Rate = 0.4

<img width="590" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/a28c0012-82c7-4bf5-a49f-643c4f66d0e3">


## TensorBoard:
After finding optimal params, we compared our 3 model types and show the results in a tensor board. 

### Model-LSTMLyricsOnly

<img width="698" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/5a3901f0-1d3b-49c4-88cf-d9b722445153">

<img width="698" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/bea07243-adef-4c46-8d5e-907dfb7c3bb7">

<img width="694" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/caa4cf53-76c6-4c1a-877a-69d0bcfc3136">

### Model-LSTMLyricsMelodies  – MelodiesFeaturesTypeT1

<img width="697" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/bf25d3ba-8d71-4511-a1d2-a91785d76074">

<img width="694" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/7dd378ca-18f5-4bca-889e-7e75f600f414">

<img width="695" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/7264fb21-468e-468b-9d93-774bf412007b">

### Model-LSTMLyricsMelodies – MelodiesFeaturesTypeT2

<img width="695" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/f905f29d-117a-457a-bf6b-42f049078300">

<img width="698" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/f0493902-e05e-467b-b9bf-adc75feb3be2">

<img width="694" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/7ec88a24-2b7d-47e1-950f-6e40d7315269">

<img width="691" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/718a1fef-62f9-43ca-9b8b-37ae7de0c743">

**Combine the results from this plot above and the plots before (of each model) we concluded that the LSTM Lyrics and Melodies model with melodies features of type T2 is the best model to choose.**

Now we will analyze the chosen model from an NLP perspective, with suitable metrics:

<img width="474" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/6adcec33-4c45-41e2-ab96-00dbaaa36ed8">

From the VADER analysis results, we can see that it did not succeed in its mission to classify songs to different sentiments. We can assume that most of the generated words are neutral, and that is the main reason for the unsuccessful sentiments' findings. Another challenge might be the length of the text – in our case song lyrics, which are significantly shorter than other text type (documents, articles etc.)

<img width="356" alt="image" src="https://github.com/ShaharBenIshay/Song-Lyrics-Generator-LSTM/assets/93884611/63954161-055c-45c6-9805-1646a0e6da7a">

In contrast to the VADER analysis results, when using TextBlog for analysis we get better results. We can see that the values gotten for each feature are close to each other between the true lyrics and generated lyrics. This means that the model that we chose has managed to capture the songs' polarity and subjectivity levels and reproduced them in the lyrics it generated.

## Lyrics Generation
Before we explain about our generating process, we set some guidelines:
* Number of paragraphs – a user can define the number of paragraphs in the generated song. We define it as 4 paragraphs.
* Number of sentences in a paragraph – a user can define the number of sentences in each paragraph. 
* Slang – we removed slang words, and we are allowing our model to generate only words that are in the word2vec model.

"Selecting Next Word” Mechanism: 

In the assignment instructions, we were asked to "select a mechanism that won’t be deterministic (for choosing the next word) but rather be sampling-based".

At first, our approach was to use a simple deterministic mechanism that just predicts the word with the highest probability. The results we had were very poor, our model always predicted the same word, and we were instructed to use a different approach. 
Then, we explored more and decided to use random.choices function which receives population (our vocabulary),  weights (probabilities), and k (how many words to choose).

This method created a sequence of random numbers corresponding to the probabilities. This means that for words with probabilities such as: "love" = 0.5, "hey" = 0.25, and "hola" = 0.25, for example, the vector of opportunities will be [love, love, hey, hola] and then there is a randomization process, and the output will be k words - in our case just 1 word. 

This approach is beneficial for two main reasons: 
1.	Increasing diversity: by sampling from the distribution of words based on their probabilities, we allow the possibility of selecting less probable but still relevant words. This increases diversity in the generated output, making it less repetitive and more interesting.
2.	Avoid repetition: deterministic approaches go for the most probable words, which can lead to the generated output becoming repetitive and lacks diversity. Sampling method helps with this issue by exploring the entire probability distribution.

