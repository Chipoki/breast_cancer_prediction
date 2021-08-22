import data_preprocessing as dp
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

    # trained logistic regression model testing
    y_prediction_lr = model_lr.predict(x_test)
    print("----------------\nLogistic Regression: \n\n", "r2 score: {}, accuracy score: {}".
          format(r2_score(y_test, y_prediction_lr), accuracy_score(y_test, y_prediction_lr)),
          '\n\n', "confusion matrix:", '\n\n', confusion_matrix(y_test, y_prediction_lr), '\n----------------\n\n')

    # trained svm model testing
    y_prediction_svm = model_svm.predict(x_test)
    print("----------------\nSVM: \n\n", "r2 score: {}, accuracy score: {}".
          format(r2_score(y_test, y_prediction_svm), accuracy_score(y_test, y_prediction_svm)), '\n\n',
          "confusion matrix:", '\n\n', confusion_matrix(y_test, y_prediction_svm), '\n----------------\n\n')

    # trained rf model testing
    y_prediction_rf = model_rf.predict(x_test)
    print("----------------\nRandom Forest: \n\n", "r2 score: {}, accuracy score: {}".
          format(r2_score(y_test, y_prediction_rf), accuracy_score(y_test, y_prediction_rf)), '\n\n',
          "confusion matrix:", '\n\n', confusion_matrix(y_test, y_prediction_rf), '\n----------------\n\n')

