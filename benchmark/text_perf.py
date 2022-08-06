import os
import os.path as osp
import time

import sklearn.ensemble
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline

from lime.lime_text import LimeTextExplainer


def interpret_data(X,
        y,
        func,
        class_names,
        newsgroups_train,
        newsgroups_test):


    explainer = LimeTextExplainer(class_names=class_names)
    times, scores = [], []
    save_dir = f"D:\\summary\\explainability\\lime\\tfidf"
    os.makedirs(save_dir, exist_ok=True)

    for r_idx in range(10):

        """
        newsgroups_test: its data field is a list of strings
        
        func: given some text, outputs a probability distribution over labels
        
        
        """

        start_time = time.time()
        exp = explainer.explain_instance(newsgroups_test.data[r_idx], func, num_features=6)
        exp.save_to_file(osp.join(save_dir, 'demo.html'))

        times.append(time.time() - start_time)
        scores.append(exp.score)
        print('...')

    return times, scores


def main():
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    test_vectors = vectorizer.transform(newsgroups_test.data)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, newsgroups_train.target)
    pred = rf.predict(test_vectors)

    f1 = sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
    print("F1", f1)

    c = make_pipeline(vectorizer, rf)

    times, scores = interpret_data(train_vectors, newsgroups_train.target, c.predict_proba, class_names,
                                   newsgroups_train, newsgroups_test)

    print(scores)


if __name__ == '__main__':
    main()
