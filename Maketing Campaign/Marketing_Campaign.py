# Pre processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
from datetime import datetime as dt

# Scoring 
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
# models 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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
            df.reset_index(drop=True, inplace=True)
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
def set_gen(data,col):
    data['Generations'] = data[col].apply(get_generation)
    return data
#

def date_to_months(df,col,year):
    current_date = dt(year, 1, 1)
    df_copy = df.copy()
    try:
        df_copy[col] = pd.to_datetime(df_copy[col])
 #       df_copy[col] = ((current_date.year - df_copy[col].dt.year) * 12) + (current_date.month - df_copy[col].dt.month)
        df_copy[col] = (current_date.year - df_copy[col].dt.year)
        # df_copy.drop(col, axis=1, inplace=True)
    except Exception as e:
        print(f"Error processing column '{col}': {e}")
        return df
    return df_copy

# Ordinal encoder function 
def preprocess_ord(df, col, cat):
    # Create an instance of the OrdinalEncoder with specific settings
    ord_enc = OrdinalEncoder(categories=[cat], encoded_missing_value=-1,
                             handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fit and transform the column data; ensure it's passed as a DataFrame slice
    ord_encoded = ord_enc.fit_transform(df[[col]])
    
    # Create a DataFrame from the encoded data, using the same column name for consistency
    encoded_df = pd.DataFrame(ord_encoded, columns=[col])
    
    # Drop the original column from df before concatenating to avoid modifying the original DataFrame
    df = df.drop(col, axis=1)
    
    # Concatenate the new encoded DataFrame with the modified original DataFrame
    df2 = pd.concat([df, encoded_df], axis=1)
    
    return df2
#
def preprocess_ohe(df, ohe_column_list):
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # List to store DataFrames of encoded columns
    encoded_dfs = []

    # Iterate over each column specified for one-hot encoding
    for column in ohe_column_list:
        # Initialize the OneHotEncoder
        ohe_encoder = OneHotEncoder(handle_unknown='error', sparse_output=False, drop='first')
        
        # Fit and transform the data of the column
        ohe_encoded = ohe_encoder.fit_transform(df_copy[[column]])
        
        # Create a DataFrame from the encoded data with appropriate column names
        ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe_encoder.get_feature_names_out(input_features=[column]))
        
        # Append the newly encoded DataFrame to the list
        encoded_dfs.append(ohe_encoded_df)
        
        # Drop the original column from the copy of the DataFrame
        df_copy.drop(column, axis=1, inplace=True)
    
    # Concatenate all the encoded DataFrames with the modified original DataFrame
    df_copy = pd.concat([df_copy] + encoded_dfs, axis=1)
    
    return df_copy

def logistic(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    lr=LogisticRegression(max_iter=50000)
    lr.fit(X_train,y_train)
    prob=lr.predict_proba(X_test)
    return (prob[:,1],y_test)

def test_train_split(df,y_value):
    X = df.drop(y_value, axis=1)
    y = df[y_value]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


# def date_to_months(df,col,current_date):
#    df[col] = pd.to_datetime(df[col])
#    # set sample date to the date of the dataset creation
#    df[col] = ((current_date.year - df[col].dt.year) * 12) + (current_date.month - df[col].dt.month)
#    df.drop(col, axis=1, inplace=True)
#    return df

def dob_gen(X_data,col):
    return


if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")