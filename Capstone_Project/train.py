
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

data_url = "https://raw.githubusercontent.com/deepankarAnand98/Udacity_MLEngineer_with_Microsoft_Azure_NDP/main/Capstone_Project/insurance.csv"
ds = TabularDatasetFactory.from_delimited_files(path=data_url)

# ds = pd.read_csv('./insurance.csv')

def clean_data(data):

    df = data.to_pandas_dataframe().dropna()
    
    df['sex'] = df.sex.apply(lambda x:0 if x=='female' else 1)
    df['smoker'] = df['smoker'].apply(lambda x: 0 if x == 'no' else 1)
    region_dummies = pd.get_dummies(df['region'], drop_first = True)
    df = pd.concat([df, region_dummies], axis = 1)
    df.drop(['region'], axis = 1, inplace = True)
    X = df.drop('charges',1)
    y = df['charges']

    X = np.array(X).astype('float32')
    y = np.array(y).astype('float32')

    return X,y


X,y = clean_data(ds)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=777)

run = Run.get_context()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=50, help="Maximum number of trees in the random forest")
    parser.add_argument('--max_depth', type=int, default=8, help="Maximum Depth of each tree")

    args = parser.parse_args()

    run.log("Total Estimators:", np.int(args.n_estimators))
    run.log("Max Depth:", np.int(args.max_depth))

    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(y_pred,y_test))
    run.log("RMSE", np.float(rmse))
    normalized_rmse = rmse/(y_test.max()-y_test.min())
    run.log("Normalized RMSE", np.float(normalized_rmse))

if __name__ == '__main__':
    main()