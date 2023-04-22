from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle as pkl
import pandas as pd
import regex as re
import numpy as np

class Data_preparation():
    def __init__(self):
        pass

    def clean_ram(self,df_train):
        # removing gb
        df_train['RAM']=df_train['RAM'].str.replace('GB','')
        df_train['RAM'] = df_train['RAM'].astype('float')
        return df_train

    def update_weights(self,df_train):
        df_train['Weight'] = df_train['Weight'].str.replace('kg','')
        df_train['Weight']=df_train['Weight'].str.replace('s','').astype(float)
        return df_train


    def cpu_cleaning(self, df_train):
        # extracting frequency
        pattern = r'(\d+(?:\.\d+)?GHz)'
        df_train['freq'] = df_train['CPU'].str.extract(pattern)
        df_train['freq'] = df_train['freq'].str.replace('GHz', '')
        df_train['freq'] = df_train['freq'].astype('float')
        df_train['CPU'] = df_train['CPU'].str.replace(pattern, '', regex=True)
        return df_train

    def manufacture_extract(self,df):
        df['cpu_manufacturer'] = df['CPU'].str.extract(r'^(\w+)')
        df['gpu_manufacturer'] = df['GPU'].str.extract(r'^(\w+)')
        return df

    def clean_storage(self,df_train):
        pattern = r'\d'
        # 1tb has 1000gb
        df_train[' Storage'] = df_train[' Storage'].str.replace('1TB', '1000GB')
        # 1tb has 1000gb
        df_train[' Storage'] = df_train[' Storage'].str.replace('2TB', '2000GB')
        df_train[' Storage'].str.extract(r'\d+(\w?\+?' ')+Flash Storage+\w')
        df_train[' Storage'] = df_train[' Storage'].str.replace('GB', '')
        df_train[' Storage2'] = df_train[' Storage'].str.replace(r' ', '')

        storage1 = []
        storage2 = []
        for i in df_train[' Storage2']:
            if len(re.findall(r'\+', i)) == 1:
                # double drive
                storage1.append(re.findall(r'(\w+)', i)[0])
                storage2.append(re.findall(r'(\w+)', i)[0])
            else:
                storage1.append(re.findall(r'(\w+)', i)[0])
                storage2.append('NaN')

        # size and type for stroage 1
        storage1type = []
        storage1size = []
        for i in storage1:
            storage1type.append(re.findall(r'(\D\w+)', i)[0])
            storage1size.append(re.findall(r'(\d+)', i)[0])

        # size and type for stroage 1
        storage2type = []
        storage2size = []
        for i in storage2:
            if i != 'NaN':
                storage2size.append(re.findall(r'(\d+)', i)[0])
                storage2type.append(re.findall(r'(\D\w+)', i)[0])
            else:
                storage2size.append(0)
                storage2type.append('NaN')

        df_train['primarystorage_size'] = storage1size
        df_train['primarystorage_type'] = storage1type
        df_train['secondarystorage_size'] = storage2size
        df_train['secondarystorage_type'] = storage2type

        df_train["primarystorage_size"] = df_train["primarystorage_size"].astype(float)
        df_train["secondarystorage_size"] = df_train["secondarystorage_size"].astype(float)
        df_train.drop(columns=[' Storage2', ' Storage'], inplace=True)

        return df_train

    def encoder(self,df):
        categorical_cols = ['Manufacturer', 'Model Name', 'Category', 'CPU',
                            'GPU', 'Operating System', 'Operating System Version',
                            'resolution', 'screentype', 'primarystorage_type',
                            'secondarystorage_type', 'cpu_manufacturer', 'gpu_manufacturer']
        en = LabelEncoder()
        for col in categorical_cols:
            df[col] = en.fit_transform(df[col])
        return df

    def clean(self,df):

        df = self.clean_ram(df)
        df = self.update_weights(df)
        df = self.cpu_cleaning(df)
        df = self.manufacture_extract(df)
        df = self.clean_storage(df)
        df.replace({'NaN': np.nan}, inplace=True)
        df = df.fillna('NaN')
        df = self.encoder(df)
        return df
#
# data_prep = Data_preparation()
#
# p = {'Manufacturer': ['Acer'], 'Model Name': ['Acer'], 'Category': ['Ultrabook'], 'Screen Size': ['15.6'], 'CPU': ['Intel Core i5 7200U 2.5GHz'], 'RAM': ['8GB'], ' Storage': ['256GB SSD'], 'GPU': ['Intel HD Graphics 620'], 'Operating System': ['Windows'], 'Operating System Version': ['10'], 'Weight': ['2.2kg'], 'resolution': ['1920x1080'], 'screentype': [None], 'touchscreen': ['1.0']}
# d = pd.DataFrame(p)
# d = data_prep.clean(d)
#
# with open("data_of_laptop.pkl", "rb") as file:
#     df = pkl.load(file)
#
#
# Y = df['Price']
# X = df.drop(columns=['Price'])
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
# dtree = DecisionTreeRegressor()
# dtree.fit(X_train, y_train)
# print(dtree.predict(d))
#
#
#
# # print(np.isnan(np.array(d)).any())
#
