# Pre processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
from datetime import datetime as dt

# Function to determine generation

def auto_drop_na(df, drop_percent):
    # Calculate the number of rows as a reference for drop calculations
    num_rows = len(df)
    drop_threshold = num_rows * drop_percent / 100  # Convert percentage to actual number of rows
    
    print(f'Drop Percent of the rows is %{drop_percent}')
    print('If the number of NA values in a column is less than the calculated threshold, automatically drop the NA rows.')

    # Get the count of NAs in each column and convert to dictionary
    has_na = df.isna().sum()
    has_na = has_na[has_na > 0].to_dict()
    print(has_na)
    
    # Make decisions about null values
    for column, na_count in has_na.items():
        if na_count > drop_threshold:
            print(f"The number of NA values in {column} is {na_count}, which is greater than the threshold {drop_threshold}. Review options before action.")
        else:
            print(f"Automatically dropping rows in {column} where NA values are present.")
            df.dropna(subset=[column], inplace=True)

    return df

def get_generation(year):
    generations = {
        range(1900, 1928): 'Lost Generation',  # Adjusted lower bound for generalization
        range(1928, 1946): 'Silent Generation',
        range(1946, 1965): 'Baby Boomers',
        range(1965, 1981): 'Generation X',
        range(1981, 1997): 'Millennials',
        range(1997, 2013): 'Generation Z',
        range(2013, 2024): 'Generation Alpha'  # Adjusted for current year inclusion
    }
    for gen_range, gen_name in generations.items():
        if year in gen_range:
            return gen_name
    return 'Unknown Generation'  # Catch-all for dates outside defined ranges
#
# Given a column with date of birth between 1900 and 2024 assign a generation
def set_gen(X_data,col):
    X_data['Generation'] = X_data[col].dt.year.apply(get_generation)
    return X_data
#
# Ordinal encoder function 
def preprocess_ord(df,col,cat):
    ord_enc = OrdinalEncoder(categories = [cat], encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
    ord_encoded = ord_enc.fit_transform(df[[col]])
    encoded_df = pd.DataFrame(ord_encoded, columns=[col])
    # df.drop(col,axis=1,inplace=True)
    df2 = pd.concat([(df.drop(col,axis=1)),encoded_df], axis=1)
    return df2
#
def X_preprocess_ohe(X_data,col):
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_encoder.fit(X_data[col].values.reshape(-1,1))
    ohe_encoded = ohe_encoder.transform(X_data[col].values.reshape(-1,1))
    ohe_encoded_df = pd.DataFrame(ohe_encoded, columns = ohe_encoder.get_feature_names_out())
    out_df  = pd.concat([ohe_encoded_df,X_data],axis=1)
    out_df.drop([col], axis=1, inplace=True)
    return out_df
#    
def test_train_split(df,y_value):
    X = df.drop(y_value, axis=1)
    y = df[y_value]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def date_to_months(df,col,current_date):
    df[col] = pd.to_datetime(df[col])
    # set sample date to the date of the dataset creation
    df[col] = ((current_date.year - df[col].dt.year) * 12) + (current_date.month - df[col].dt.month)
    df.drop(col, axis=1, inplace=True)
    return df

def dob_gen(X_data,col):
    return


if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")