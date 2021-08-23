from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import numpy as np


class Summarizer:

    def __init__(self, data_preprocessor):
        self.data_preprocessor = data_preprocessor

    def summary_printer(self, models):
        for model in models:
            x_train, x_test, y_train, y_test = self.data_preprocessor.get_processed_data()
            y_prediction = model.predict(x_test)
            k_fold_accuracies = cross_val_score(model, x_train, y_train, cv=10)
            print("----------------\n{}: \n\n".format(type(model).__name__), "r2 score: {}, accuracy score: {}, \n"
                                                                             " k fold mean accuracy: {}, k fold std: {}"
                  .format(r2_score(y_test, y_prediction), accuracy_score(y_test, y_prediction),
                          np.mean(k_fold_accuracies), np.std(k_fold_accuracies)),
                  '\n\n', "confusion matrix:", '\n\n', confusion_matrix(y_test, y_prediction), '\n----------------\n\n')
