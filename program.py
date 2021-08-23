import data_preprocessing as dp
import summarizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

if __name__ == '__main__':

    # data preprocessing
    data = pd.read_csv("./data/breast-cancer-wisconsin.csv", na_values='?')
    data_preprocessor = dp.DataPreprocessing(data)
    data_preprocessor.impute_missing_data(missing_val=np.nan)
    # data_preprocessor.encode_dependent_data()
    data_preprocessor.split_data(test_size=0.20)

    # data acquisition
    x_train, x_test, y_train, y_test = data_preprocessor.get_processed_data()

    # logistic regression training
    model_lr = LogisticRegression(random_state=0)
    model_lr.fit(x_train, y_train)

    # svm classification model training
    model_svm = SVC()
    model_svm.fit(x_train, y_train)

    # random forest model training
    model_rf = RandomForestClassifier()
    model_rf.fit(x_train, y_train)

    # trained models testing
    summarizer = summarizer.Summarizer(data_preprocessor)
    models = [model_lr, model_svm, model_rf]
    summarizer.summary_printer(models)

