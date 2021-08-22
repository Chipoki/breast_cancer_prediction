import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer


class DataPreprocessing:

    def __init__(self, data):
        self.data = data
        self.x_train = np.array
        self.x_test = np.array
        self.y_train = np.array
        self.y_test = np.array
        self.x, self.y = self.organize_data()

    def organize_data(self):
        x = self.data.iloc[:, 1:-1].values
        y = self.data.iloc[:, -1].values
        return x, y

    def impute_missing_data(self, missing_val):
        imputer = KNNImputer(missing_values=missing_val)
        imputer.fit(self.x)
        self.x = imputer.transform(self.x)

    def encode_independent_data(self, column):
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [column])], remainder='passthrough')
        self.x = np.array(ct.fit_transform(self.x))

    def encode_dependent_data(self):
        le = LabelEncoder()
        self.y = np.array(le.fit_transform(self.y))

    def split_data(self, test_size):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size,
                                                                                random_state=0)

    def rescaled_data(self):
        sc = StandardScaler()
        x_train_scaled = sc.fit_transform(self.x_train)
        x_test_scaled = sc.transform(self.x_test)
        return x_train_scaled, x_test_scaled

    def get_processed_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

