# NER
A lightweight, trainable, Keras-based Named Entity Recognizer

A model that accepts a named entity recognition (NER) training file in `.iob` format and trains an LSTM model to predict the IOB tags for each token of each sentence. All sentences are pre-padded to be of the same length. Uses an internal validation set to train with early stopping and also evaluates model performance on a test set. The weights and configuration of the trained model are also saved out.

This repo currently trains on the [MIT trivia10k13 Movie Corpus](https://groups.csail.mit.edu/sls/downloads/).
