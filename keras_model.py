import random
import math

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

def my_accuracy(y_true, y_pred):
    print(y_true)
    print(y_pred)

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Dataset:
    def __init__(self, path: str) -> None:
        self.samples = []
        labels = set()
        vocab = set()
        text = open(path).read()
        for raw_query in text.split("\n\n"):
            words = []
            entities = []
            for line in raw_query.split("\n"):
                if len(line) == 0:
                    continue
                word, entity = line.split("\t")

                words.append(word)
                entities.append(entity)

                labels.add(entity)
                vocab.add(word)

            self.samples.append((words, entities))

        vocab = sorted(list(vocab))
        vocab = ["<NULL>"] + vocab
        self.vocab_size = len(vocab)
        print("vocab size:", self.vocab_size)
        self.word2i = {word: i for i, word in enumerate(vocab)}
        self.i2word = {i: word for i, word in enumerate(vocab)}

        labels = sorted(list(labels))
        # Move 'O' to the beginning, so it's index is 0 (needed when
        # padding).
        labels = ["O"] + [lbl for lbl in labels if lbl != "O"]
        self.num_labels = len(labels)
        print("num labels:", self.num_labels)
        self.label2i = {lbl: i for i, lbl in enumerate(labels)}
        self.i2label = {i: lbl for i, lbl in enumerate(labels)}

        random.shuffle(self.samples)
    
    @property
    def Xy(self, test_ratio: float = 0.25) -> tuple:
        X = []
        y = []

        for words, entities in self.samples:
            X.append([self.word2i[w] for w in words])
            y.append([self.label2i[lbl] for lbl in entities])

        train_n = math.ceil(len(X)*(1-test_ratio))

        trainx = pad_sequences(X[:train_n])
        trainy = pad_sequences(y[:train_n])
        testx = pad_sequences(X[train_n:])
        testy = pad_sequences(y[train_n:])

        return trainx, trainy, testx, testy


class NERModel:
    def __init__(self, vocab_size: int, num_labels: int, hidden_size: int = 128, dropout: float = 0.2) -> None:
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, hidden_size))
        self.model.add(LSTM(hidden_size, dropout=dropout, return_sequences=True))
        self.model.add(Dense(num_labels, activation="softmax"))
    
    def fit(self, data: Dataset) -> None:
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
        )
        X, y, _, _ = data.Xy
        self.model.fit(X, y, epochs=100, validation_split=0.25, callbacks=[EarlyStopping()])
    
    def test(self, data: Dataset) -> None:
        _, _, X, y = data.Xy
        self.model.evaluate(X, y)
        n_sequences, sequence_len = X.shape
        predictions = self.model.predict(X)
        flat_predictions = []
        flat_gold = []
        with open("predictions.txt", "w") as f:
            f.write(f"WORD\tPREDICTION\tGOLD STANDARD\n")
            for sequence_i in range(n_sequences):
                f.write("\n")
                for word_i in range(sequence_len):
                    word = data.i2word[X[sequence_i, word_i]]
                    prediction_i = np.argmax(predictions[sequence_i, word_i, :])
                    prediction = data.i2label[prediction_i]
                    flat_predictions.append(prediction)
                    truth = data.i2label[y[sequence_i, word_i]]
                    flat_gold.append(truth)
                    f.write(f"{word}\t{prediction}\t{truth}\n")
        with open("metric-report.txt", "w") as f:
            report = classification_report(flat_gold, flat_predictions)
            f.write(report)
    
    def save(self) -> None:
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

if __name__ == "__main__":
    train_data = Dataset("movie_query_train.iob")
    model = NERModel(train_data.vocab_size, train_data.num_labels)
    model.fit(train_data)
    model.test(train_data)
    model.save()