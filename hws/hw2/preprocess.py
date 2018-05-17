import pandas as pd
import numpy as np

def null_cols(data_frame):
    #return data_frame.columns[data_frame.isnull().any()]

    isnull = data_frame.isnull().any()
    isnull_cols = list(isnull[isnull == True].index)

    return isnull_cols


def fill_missing_median(df):
    '''
    Fills all missing values with the median value of the given column
    Inputs:
        data_frame (pandas dataframe)
    Returns:
        data_frame (pandas dataframe) with missing values imputed
    '''

    nc = null_cols(df)
    
    #for col in nc:
    #    print(col)
    #    col_med = data_frame[col].median()
    #     print(col_med)
    #    data_frame[col].replace('NaN',col_med).fillna(col_med, inplace = True)


    for col in nc:
        col_median = df[col].median()
        print(col_median)
        df[col].fillna(col_median, inplace = True)

        return df

    #return data_frame

def fill_missing_mean(data_frame):
    '''
    Fills all missing values with the mean value of the given column
    Inputs:
        data_frame (pandas dataframe)
    Returns:
        data_frame (pandas dataframe) with missing values imputed
    '''
    return data_frame.fillna(data_frame.mean(), inplace=True)

def drop_vars(data_frame, var_list_to_drop):
	'''
	Drops identified variables from dataframe
	Inputs:
		data_frame (pandas dataframe)
		var_list_to_drop (list): list of variables to drop
	Returns:
		data_frame (pandas dataframe)
	'''
	return data_frame.drop(var_list_to_drop, axis=1)
