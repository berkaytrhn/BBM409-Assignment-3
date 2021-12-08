import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayesClassifier
from tf_idf import TfIdf
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def main(args):
    # read data

    data = pd.read_csv("emails.csv")

    data = np.array(data)

    # first 5 sample
    #temp = np.vstack((np.array(data[data.iloc[:,-1] == 1].iloc[:5,:]), np.array(data[data.iloc[:,-1] == 0].iloc[:5,:])))

    #print(temp.shape)


    X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1].astype(np.int32), test_size=0.2, shuffle=True)

    
    # remove 'subject:' text from data
    X_train = preprocess_text(X_train)
    X_test = preprocess_text(X_test)
    

    nbc = NaiveBayesClassifier()
    nbc.fit(X_train, y_train, n_number=1)
    predicted = nbc.predict(X_test)
    print(predicted.shape, predicted, predicted.dtype)
    print(y_test.shape, y_test, y_test.dtype)
    print(f"Accuracy: {round(accuracy_score(y_test, predicted)*100, 4)}%")

    tfIdfC = TfIdf()
    tfIdfC.fit(X_train, y_train)
    predicted = tfIdfC.predict(X_test)
    print(predicted.shape, predicted, predicted.dtype)
    print(y_test.shape, y_test, y_test.dtype)
    print(f"Accuracy: {round(accuracy_score(y_test, predicted)*100, 4)}%")

    # print(X_train[0:2])
    # val, val2 = tf_idf_generator(X_train)
    # print(val2)
    # print(val)

    # precision = precision_score(y_test, predicted)
    # recall = recall_score(y_test, predicted)
    # f1 = f1_score(y_test, predicted)
    # accuracy = accuracy_score(y_test, predicted)
    # _confusion_matrix = confusion_matrix(y_test, predicted)
    # display_confusion_matrix(_confusion_matrix)
    # print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--temp", type=str, required=False)


    args = parser.parse_args()

    main(args)