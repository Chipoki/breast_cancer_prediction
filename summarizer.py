from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


class Summarizer:

    def __init__(self, data_preprocessor):
        self.data_preprocessor = data_preprocessor

    def model_testing_printer(self, models):

        kfold_accuracies, kfold_stds = {}, {}

        for model in models:
            x_train, x_test, y_train, y_test = self.data_preprocessor.get_processed_data()
            y_prediction = model.predict(x_test)
            k_fold_accuracies = cross_val_score(model, x_train, y_train, cv=10)
            kfold_accuracy = np.mean(k_fold_accuracies)
            kfold_std = np.std(k_fold_accuracies)
            model_name = type(model).__name__
            kfold_accuracies[model_name] = kfold_accuracy
            kfold_stds[model_name] = kfold_std

            print("----------------\n{}: \n\n".format(model_name), "accuracy score: {}, \n"
                                                                   " k fold mean accuracy: {}, k fold std: {}"
                  .format(accuracy_score(y_test, y_prediction), kfold_accuracy, kfold_std), '\n\n',
                  "confusion matrix:", '\n\n', confusion_matrix(y_test, y_prediction), '\n----------------\n\n')

        self.summary_printer(kfold_accuracies, kfold_stds)

    @staticmethod
    def summary_printer(kfold_accuracies, kfold_stds):

        maximal_accuracy_model = max(kfold_accuracies, key=lambda key: kfold_accuracies[key])
        minimal_std_model = min(kfold_stds, key=lambda key: kfold_stds[key])
        accuracy_difference = (kfold_accuracies[maximal_accuracy_model] - kfold_accuracies[minimal_std_model])
        std_difference = (kfold_stds[maximal_accuracy_model] - kfold_stds[minimal_std_model])

        print("the most accurate model: {}, \nthe model with the lowest std of kfold accuracies: {}".format(
            maximal_accuracy_model, minimal_std_model),
            "\n\nBetween those models, The difference in accuracy is: {}\nand the difference in std is: {} ".format(
                accuracy_difference, std_difference))
