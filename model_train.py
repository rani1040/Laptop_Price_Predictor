from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import pickle as pkl
from  data_preparation import Data_preparation

class Prediction():
    def __init__(self):
        pass

    def training(self):
        with open("data_of_laptop.pkl", "rb") as file:
            df = pkl.load(file)
        Y = df['Price']
        X = df.drop(columns=['Price'])
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
        dtree = DecisionTreeRegressor()
        dtree.fit(X_train, y_train)
        return dtree
    def predict_answer(self,p):
        query_point = pd.DataFrame(p)
        data_prep = Data_preparation()
        query_point = data_prep.clean(query_point)
        d_tree = self.training()
        return d_tree.predict(query_point)

# p = {'Manufacturer': ['Acer'], 'Model Name': ['Acer'], 'Category': ['Ultrabook'], 'Screen Size': ['15.6'], 'CPU': ['Intel Core i5 7200U 2.5GHz'], 'RAM': ['8GB'], ' Storage': ['256GB SSD'], 'GPU': ['Intel HD Graphics 620'], 'Operating System': ['Windows'], 'Operating System Version': ['10'], 'Weight': ['2.2kg'], 'resolution': ['1920x1080'], 'screentype': [None], 'touchscreen': ['1.0']}






