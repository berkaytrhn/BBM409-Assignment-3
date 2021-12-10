import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns

def bag_of_words_ngram(data, n_number=1):
    # data parameter -> array of text samples: [text1, text2, ...]
    vct = CountVectorizer(ngram_range=(n_number, n_number))
    ngram = vct.fit_transform(data)
    return ngram, vct.get_feature_names_out()

def tf_idf_generator(data):
    # data parameter -> array of text samples: [text1, text2, ...]
    tf_idf = TfidfVectorizer(min_df=1)
    tf_idf_vals = tf_idf.fit_transform(data)
    return tf_idf_vals, tf_idf_vals.toarray(), tf_idf.get_feature_names_out()

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
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def print_topN(bow_applied_2,feature_names):
    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(bow_applied_2.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    print("\n===Keywords===")
    for k in keywords:
        print(k, keywords[k])