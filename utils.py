import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns

def bag_of_words_ngram(data, n_number=1):
    # data parameter -> array of text samples: [text1, text2, ...]
    vct = CountVectorizer(ngram_range=(n_number, n_number))
    ngram = vct.fit_transform(data).toarray()
    return ngram, vct.get_feature_names_out()

def tf_idf_generator(data):
    # data parameter -> array of text samples: [text1, text2, ...]
    tf_idf = TfidfVectorizer(min_df=0)
    tf_idf_vals = tf_idf.fit_transform(data).toarray()
    return tf_idf_vals, tf_idf.get_feature_names_out()

def preprocess_text(data):
    iterable = map(lambda x:" ".join(x[0].split(":")[1:]) , data)
    result = np.array(pd.DataFrame(np.array([np.array(text) for text in iterable])).iloc[:,0].str.replace("[^\w\s]",""))
    return result


def display_confusion_matrix(confusion_matrix):
    # heatmap display for confusion matrix
    labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    length = len(max(labels)) + 4
    labels = np.asarray(labels).reshape(2, 2)

    annots = [f"{str(label)}({str(value)})" for array in np.dstack((labels, confusion_matrix)) for (label, value) in
              array]
    annots = np.asarray(annots).reshape(2, 2).astype(str)
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=annots, cmap="YlGnBu", fmt=f".{length}")
    plt.show()
