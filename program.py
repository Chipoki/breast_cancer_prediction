import data_preprocessing as dp
import summarizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    # data preprocessing
    data = pd.read_csv("./data/breast-cancer-wisconsin.csv", na_values='?')
    data_preprocessor = dp.DataPreprocessing(data)
    data_preprocessor.impute_missing_data(missing_val=np.nan)
    # data_preprocessor.encode_dependent_data()
    data_preprocessor.split_data(test_size=0.20)

    # data acquisition
    x_train, x_test, y_train, y_test = data_preprocessor.get_processed_data()

    # logistic regression model training
    model_lr = LogisticRegression(random_state=0)
    model_lr.fit(x_train, y_train)

    # svm classification model training
    model_svm = SVC(kernel="rbf", random_state=0)
    model_svm.fit(x_train, y_train)

    # decision tree model training
    model_dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
    model_dt.fit(x_train, y_train)

    # random forest model training
    model_rf = RandomForestClassifier(criterion="entropy", random_state=0)
    model_rf.fit(x_train, y_train)

    # naive bayes model training
    model_nb = GaussianNB()
    model_nb.fit(x_train, y_train)

    # knn model training
    model_knn = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
    model_knn.fit(x_train, y_train)

    # trained models testing
    summarizer = summarizer.Summarizer(data_preprocessor)
    models = [model_lr, model_svm, model_dt, model_rf, model_knn, model_nb]
    summarizer.model_testing_printer(models)

