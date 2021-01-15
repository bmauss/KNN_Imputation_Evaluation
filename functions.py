import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


def cat_codes(df, columns):
    """
    Input: Data frame and list of columns
    Output: Columns converted to categories and assigned cat_codes
    """
    for i in columns:
        df[i] = df[i].astype('category')
        df[i] = df[i].cat.codes
        
def knn_continuous(df, target, frac_nan=0.1, n_neighbors=5, seed=None):
    '''
    Input:
    df: Pandas dataframe
    target: column name for the dataset's target variable.
    frac_nan: float between 0 and 1. Fraction of data to remove from dataframe (Default: 10%)
    n_neighbors: integer. Number of neighbors for KNN to consider (Default: 5)
    seed: integer. Set random seed for reproducibility (Default: None)
    
    Output: Report of the number of values removed and ratio to dataset. Report of the number of Perfect Estimations. RMSE of KNN Imputation.
    '''
    # Define feature columns
    feat_cols = df.drop(target, axis=1)
    features = list(feat_cols.columns)
    
    # Label encode target variable
    encoded_df = cat_codes(df, list(target))
    
    # Instantiate Scaler
    scaler = MinMaxScaler()

    scaled_df = pd.DataFrame(scaler.fit_transform(encoded_df), 
                           columns=encoded_df.columns)
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Inserting NaN values into Experiment Group
    for col in scaled_df[features]:
        scaled_df.loc[scaled_df.sample(frac=frac_nan, 
                                           replace=True).index,
                                              col] = np.nan
    
    # Total number of feature values 
    num_vals = len(scaled_df.index) * len(features)
    print(f'The dataset (without target) has a total of {num_vals} values')

    # Calculate number of NaNs
    num_nan = scaled_df.isna().sum().sum()
    print(f'There are {num_nan} NaN values')

    # Percent of missing values
    percent_nan = (num_nan / num_vals) * 100
    print(f'{round(percent_nan, 2)}% of the dataset is missing')

    # Calculate number of rows with missing values

    # obtaining indices of rows with NaN values
    nan_cols = scaled_df[features]
    nan_cols = nan_cols[nan_cols.isna().any(axis=1)]
    nan_rows = len(nan_cols.index)
    print(f'There are {nan_rows} rows with missing values')

    # Percentage of entries with missing data
    total_missing = (nan_rows / len(scaled_df.index)) * 100
    print(f'{round(total_missing, 2)}% of the rows contain missing values')
    print('\n')
    print('-----------------------------------------------------------------')
    print('\n')
    
    # Creating list of indices 
    null_idx = list(nan_cols.index)

    # Answer Key
    answer_key = encoded_df.iloc[null_idx]
    
    #Instantiate KNNImputer
    impute = KNNImputer(n_neighbors = 5)

    # Applying to dataframe
    knn_df = pd.DataFrame(impute.fit_transform(scaled_df), 
                           columns=scaled_df.columns)
    
    # Inverting Scaling
    inverse_knn_df = pd.DataFrame(scaler.inverse_transform(knn_df), 
                               columns=knn_df.columns)
    
    # Subsetting data to match that of our answer key
    test_df = inverse_knn_df.iloc[null_idx]
    
    # Resetting indexes of test_df and answer_key for iteration
    test_df = test_iris.reset_index()
    test_df.drop(['index', target], axis=1, inplace=True)
    answer_key = answer_key.reset_index()
    answer_key.drop(['index', target], axis=1, inplace=True)

    # Calculate results
    results = pd.DataFrame((round((answer_key - test_df), 3)))
    
    # Imputes where y - y_hat != 0
    imperfect_imputes = 0

    for col in results.columns:
        for i in range(len(results)):
            if results[col][i] != 0.00 or results[col][i] != -0.00:
            imperfect_imputes += 1

    # Imputes where y - y_hat == 0
    perfect_imputes = num_nan - imperfect_imputes

    print(f'Total Values Imputed: {total_imputes}')
    print(f'Imperfect Imputations: {imperfect_imputes}')
    print(f'Perfect Imputations: {perfect_imputes}')
    print('\n')
    print('-----------------------------------------------------------------')
    print('\n')
    
    # Calculate and store squared errors
    squared_terms = []
    for col in results.columns:
        for i in range(len(results)):
            if results[col][i] != 0.00 or results[col][i] != -0.00:
                error = results[col][i]
                squared_error = error**2
                squared_terms.append(squared_error)

    # Calculate sum of squared errors
    sum_sqr_err = sum(squared_terms)
    
    # Calculate MSE
    mse = sum_sqr_err/num_nan
    
    # Calculate RMSE
    rmse = np.sqrt(mse)

    print(f'RMSE for KNN Imputation on dataset is {rmse}')

    
    
    
    
   
    
    