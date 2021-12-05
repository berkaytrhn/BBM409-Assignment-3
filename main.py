import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayesClassifier
from utils import *


def main(args):
    # read data

    data = pd.read_csv("emails.csv")
    print(data.head())

    # first 5 sample
    temp = np.vstack((np.array(data[data.iloc[:,-1] == 1].iloc[:5,:]), np.array(data[data.iloc[:,-1] == 0].iloc[:5,:])))

    print(temp.shape)

    X_train, X_test, y_train, y_test = train_test_split(temp[:,0], temp[:,1], test_size=0.2, shuffle=True)

    
    # remove 'subject:' text from data
    X_train = preprocess_text(X_train)
    X_test = preprocess_text(X_test)


    nbc = NaiveBayesClassifier()
    nbc.fit(X_train, y_train, n_number=1)
    nbc.predict(X_test)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--temp", type=str, required=False)


    args = parser.parse_args()

    main(args)