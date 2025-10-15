import gzip
import pickle
import pandas as pd
import json

def wrangle(filename):
    # Load the ZIP compressed CSV file with latin-1 encoding
    df = pd.read_csv(filename, compression='zip', encoding='latin-1')
    return df

def make_predictions(data_filepath, model_filepath):
    # Wrangle compressed csv file with latin-1 encoding
    X_test = wrangle(data_filepath)
    # Load model
    with open(model_filepath,"rb") as f:
        model=pickle.load(f)
    # Generate predictions
    y_test_pred = model.predict(X_test)
    # Put predictions into Series with name "Result", and same index as X_test
    y_test_pred = pd.Series(y_test_pred,index=X_test.index,name="Result")
    return y_test_pred