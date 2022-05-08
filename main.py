from statistics import mean

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
import nltk as nl
from sklearn.feature_extraction import _stop_words
from numpy import std
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
import nltk
from nltk.stem import PorterStemmer

porter_stemmer = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('stopwords')


def showResult():
    spam_data = pd.read_csv('./Dataset/spam.csv', delimiter=',', encoding='latin-1')
    real_data = pd.read_csv('./Dataset/real.csv', delimiter=',', encoding='latin-1')
    combined_data = showDataSet(spam_data, real_data)
    dataPreprocessing(combined_data) #data pre-processing for combined sms data

    real_data = real_data.rename(columns={'type': 'label', 'text': 'message'})
    real_data.head()
    print("Real Data:")
    print(real_data.describe())
    print(real_data[real_data['label'] == 'ham'].count().get(0))
    print(real_data[real_data['label'] == 'spam'].count().get(0))
    dataPreprocessing(real_data) #data pre-processing for real sms data

    X = combined_data.message
    Y = combined_data.label
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)

    X_real = real_data.message
    Y_real = real_data.label
    Y_real = le.fit_transform(Y_real)
    Y_real = Y_real.reshape(-1, 1)

    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    X_transform = sequence.pad_sequences(sequences, maxlen=max_len)

    tok.fit_on_texts(X_real)
    sequences = tok.texts_to_sequences(X_real)
    X_transform_real = sequence.pad_sequences(sequences, maxlen=max_len)

    validation_ratio = 0.15
    test_ratio = 0.15
    test_ratio_real = 0.30

    # SPLIT DATASET
    X_train, X_test, Y_train, Y_test = train_test_split(X_transform, Y, test_size=test_ratio)

    X_train_real, X_test_real, Y_train_real, Y_test_real = train_test_split(X_transform_real, Y_real,
                                                                            test_size=test_ratio_real)

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train,
                                                      test_size=test_ratio / (test_ratio + validation_ratio),
                                                      random_state=1)


    print("\n ****Naive Bayes:***** \n")
    nb = MultinomialNB()
    nb.fit(X_train, Y_train.reshape(len(Y_train), ))
    nb_pred = nb.predict(X_test)
    evaluationResult(Y_test.reshape(len(Y_test), ), nb_pred, 'naive bayes')

    print("\n ****Naive Bayes Real Data Result*****: \n")
    nb_real = MultinomialNB()
    nb_real.fit(X_train_real, Y_train_real.reshape(len(Y_train_real), ))
    nb_pred_real = nb_real.predict(X_test_real)
    evaluationResult(Y_test_real.reshape(len(Y_test_real), ), nb_pred_real, 'naive bayes real')

    scores_nb = cross_val_score(nb, x_val, y_val, scoring='accuracy', cv=5, n_jobs=-1)
    print('K-fold Accuracy Naive Bayesian : %.3f (%.3f)' % (mean(scores_nb), std(scores_nb)))
    showWrongLabeledMessages(X_real, X_test_real, Y_test_real, nb_pred_real)

    print("\n ****SVM:***** \n")
    svm = SVC()
    svm.fit(X_train, Y_train.reshape(len(Y_train), ))
    svm_pred = svm.predict(X_test)
    evaluationResult(Y_test.reshape(len(Y_test), ), svm_pred, 'SVM')

    print("\n ****SVM Real Data Result*****: \n")
    svm_real = SVC()
    svm_real.fit(X_train_real, Y_train_real.reshape(len(Y_train_real), ))
    svm_pred_real = svm_real.predict(X_test_real)
    evaluationResult(Y_test_real.reshape(len(Y_test_real), ), svm_pred_real, 'SVM Real')

    scores_svm = cross_val_score(svm, x_val, y_val, scoring='accuracy', cv=5, n_jobs=-1)
    print('K-fold Accuracy SVM : %.3f (%.3f)' % (mean(scores_svm), std(scores_svm)))
    showWrongLabeledMessages(X_real, X_test_real, Y_test_real, svm_pred_real)



    print("\n ****KNN: \n")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train.reshape(len(Y_train), ))

    knn_pred = knn.predict(X_test)
    evaluationResult(Y_test.reshape(len(Y_test), ), knn_pred, 'KNN')

    print("\n ****KNN Real Data Result********: \n")
    knn_real = KNeighborsClassifier(n_neighbors=3)
    knn_real.fit(X_train_real, Y_train_real.reshape(len(Y_train_real), ))
    knn_pred_real = knn.predict(X_test_real)
    evaluationResult(Y_test_real.reshape(len(Y_test_real), ), knn_pred_real, 'KNN Real')

    scores_knn = cross_val_score(knn, x_val, y_val, scoring='accuracy', cv=5, n_jobs=-1)
    print('K-fold Accuracy KNN : %.3f (%.3f)' % (mean(scores_knn), std(scores_knn)))
    showWrongLabeledMessages(X_real, X_transform_real, Y_real, knn_pred_real)


def dataPreprocessing(df):
    # All stop words combined for preprocessing
    nltk_stopwords = nl.corpus.stopwords.words('english')
    gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS
    sklearn_stopwords = _stop_words.ENGLISH_STOP_WORDS
    combined_stopwords = sklearn_stopwords.union(nltk_stopwords, gensim_stopwords)
    # preprocessing on sms_dataset
    df['message'] = df['message'].apply(lambda x: x.lower())
    df['message'] = df['message'].str.replace('[^\w\s]', '')
    df['message'] = df['message'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (combined_stopwords)]))



def showDataSet(data1, data2):
    df = data1.append(data2)
    df = df.rename(columns={'type': 'label', 'text': 'message'})
    df.head()
    print(df.describe())  # combined datas
    check = 'ham'
    print("Ham data count:")
    print(df[df['label'] == check].count().get(0))
    check = 'spam'
    print("Spam data count:")
    print(df[df['label'] == check].count().get(0))

    return df


def showWrongLabeledMessages(X, X_test, Y_test, pred):
    for row_index, (input, prediction, label) in enumerate(zip(X_test, pred, Y_test)):
        if prediction != label[0]:
            print('Message :', X[row_index], 'has been classified as ', prediction, 'and should be ',
                  label[0])


def evaluationResult(y_test, pred, name):
    print(classification_report(y_test, pred, target_names=['Ham', 'Spam']))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, pred),
                       columns=['Predicted Ham', 'Predicted Spam'],
                       index=['Ham', 'Spam']))

    print(f'Accuracy: {round(accuracy_score(y_test, pred), 5)}')

    plt.figure(figsize=(10, 4))

    heatmap = sns.heatmap(data=pd.DataFrame(confusion_matrix(y_test, pred)), annot=True, fmt="d",
                          cmap=sns.color_palette("Blues", 50))
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('Ground Truth for ' + name)
    plt.xlabel('Prediction for ' + name)
    plt.show()

    precision, recall, thresholds = precision_recall_curve(y_test, pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label='Model')
    ax.set_xlabel('Recall for ' + name)
    ax.set_ylabel('Precision for ' + name)
    ax.legend(loc='center left')


if __name__ == '__main__':
    showResult()
