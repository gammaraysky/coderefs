import numpy as np
import pandas as pd
from typing import List, Dict
from collections import Counter
import pyodbc
import os
import re
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,\
                                  StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


##### LOADING/SAVING DATA #####
def sql_query_to_csv(server, database, username,
                    password, driver, query, data_path) -> None:
    """Reads in dataset from SQL, merge tables and save to CSV

    Example:
    server = 'aiap-training.database.windows.net'
    database = 'aiap'
    username = 'apprentice'
    password = '{Pa55w.rd}'
    driver = '{ODBC Driver 17 for SQL Server}'
    data_path = './data/raw/data.csv'
    query = 'SELECT * FROM transactions \
        LEFT JOIN flats on transactions.id = flats.id \
        LEFT JOIN towns on transactions.townid = towns.townid'
    """
    with pyodbc.connect(
            'DRIVER='+driver+
            ';SERVER=tcp:'+server+
            ';PORT=1433;DATABASE='+database+
            ';UID='+username+
            ';PWD='+ password) as conn:
        df = pd.read_sql(query, conn, index_col=None)

    save_csv(df, data_path)
    return None

def save_csv(df, data_path) -> None:
    """Saves DataFrame to supplied data_path as CSV"""
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))
    df.to_csv(data_path, index=False)
    return None

def load_csv(data_path) -> pd.DataFrame:
    """Loads CSV from specified path as a DataFrame"""
    df = pd.read_csv(data_path)
    return df
##########################





##### MISSING VALUES #####
def missing_rows_per_col(df) -> None:
    """Prints out number of missing values for
    each column of a given DataFrame.
    """
    print('COLUMN                   ROWS MISSING           %')
    for col in df.columns:
        rows_missing = df[col].isna().sum()
        print(
            f"{col:<30} {rows_missing:>6,.0f}      {rows_missing/len(df):>6.1%}")

def list_missing_indices(s:pd.Series) -> List:
    """Returns a list of indices in a Series where isna()"""
    return list(s[s.isna()].index)

def rows_n_missing_counts(
        df:pd.DataFrame,
        ncols:int|None=None) -> pd.Series|List:
    """Get counts of number of missing columns for any indices with missing
    values.

    If ncols is specified, returns a list of indices with n or more columns
    missing. Otherwise, returns a Series with indices and their corresponding
    counts of missing fields.

    :params
    df - DataFrame to be checked

    ncols - int|None
        If specified, returns only list of indices where n or more columns
        have missing values.

        If unspecified, returns Series of indices and respective counts,
        sorted largest first.
    """
    indices_missing_per_col = []
    [indices_missing_per_col.append(
        list_missing_indices(df[col])) for col in df.columns]
    flatlist = [
        item for sublist in indices_missing_per_col for item in sublist]
    counts = Counter(flatlist)
    counts = pd.Series(counts)

    if ncols:
        return list(counts[counts>=ncols].index)

    return counts.sort_values(ascending=False)

def rows_n_missing_report(
        df:pd.DataFrame,
        ncols:int|None=None) -> None:
    """

    """
    counts = rows_n_missing_counts(df, ncols)
    values = list(counts.unique())
    for v in values:
        print('# rows with {} columns or more missing: {} ({:.3%})'.format(
            v,
            len(counts[counts>=v].index),
            len(counts[counts>=v].index)/len(df),
        ))
##########################


##### CLEANING #####
def regex_extract_numbers(row:str) -> float:
    """This sample method expects the following input values:
    '47'
    '47 years'
    '47 years 6 months'
    and outputs years in float. e.g. 47.5

    Typical Usage:
    series = series.apply(regex_extract_numbers)
    """
    pattern = r'(\d+)'
    matches = re.findall(pattern, row)
    if len(matches)>1:
        return float(matches[0]) + float(matches[1])/12
    else:
        return float(matches[0])
##########################


##### EDA SUMMARY STATS AND PLOTS #####
def univariate_reports(
        df, cols, bins,
        nrows, ncols, figsize) -> None:
    """Print summary stats and plots histogram/boxplot/countplots for the
    given columns of a DataFrame.
    """
    display(df[cols].describe())

    plt.rc('xtick', labelsize=7)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=figsize, constrained_layout=True)

    for col, ax in zip(cols, axs.ravel()):
        if df[col].dtype=='float64' or df[col].dtype=='int64':
            ax2 = ax.twinx()
            ax.hist(df[col], bins=bins)
            ax2.boxplot(df[col], vert=False, whis=1.5)
        else:
            sns.countplot(x=df[col], ax=ax)
            print(df[col].value_counts())
            print()
        ax.set_title(col)
        for item in ax.xaxis.get_ticklabels():
            item.set_rotation(90)

    plt.show()
##########################


##### FEATURE TRANSFORMATIONS #####

def column_quantile(series:pd.Series, quantiles:List|int,
                    duplicates:str='drop') -> pd.Series:
    """Transforms quantitative column into quantiles, and casts into int
    values.

    Use for quantitative data that do not fit Gaussian or Power Law
    distributions.

    : params
    series:pd.Series
        series to cut into quantiles
    quantiles:List|int
        List - e.g. [0, 0.25, 0.5, 0.75, 1] will cut values into 4 quartiles.
        int - e.g. 4 will cut values into 4 quartiles.
    duplicates:"raise" | "drop" (default)
        how to handle if duplicate bin edges are found

    : return
        Series of ints. e.g.
        If input was test scores ranging from 0-100, and was cut into 4
        quartiles, the series output would be [0, 1, 2, 3, 1, 0...]
    """
    result = pd.qcut(series, q=quantiles,
                     labels=False, duplicates=duplicates).astype('int64')
    return result


def column_bin(series:pd.Series, bins:List|int,
               duplicates:str='drop') -> pd.Series:
    """Transforms quantitative column into bins, and casts into int values.

    : params
    series:pd.Series
        series to cut into bins
    bins:List|int
        List - e.g. [0, 10, 20, 30, 40]
        int - e.g. 4 automatically find data range and split into 4 equal
        intervals
    duplicates:"raise" | "drop" (default)
        how to handle if duplicate bin edges are found

    : return
        Series of ints.
    """
    result = pd.cut(series, bins, include_lowest=True,
                    labels=False,duplicates=duplicates).astype('int64')
    return result


def column_logtransform(df:pd.DataFrame, cols:List) -> None:
    """Applies log transformation on quantitative columns.
    Use for quantitative data that fits a Power Law distribution.

    : params
    df:pd.DataFrame
        input DataFrame with columns to transform. new columns with '_log'
        suffix will be added in place.
    cols:List
        List of column names in the DataFrame to be transformed
    : return
    None
        New log-transformed columns are appended in place to the existing
        DataFrame.
    """
    for c in cols:
        c_log = c+'_log'
        df[c_log] = [np.log(i)  if i>0 else 0 for i in df[c]]
    return

def column_ordinalmap(series:pd.Series, mapping:Dict) -> pd.Series:
    """Map nominal categorical data to ordinal.

    e.g. Education field with the following text values, to map into years
    spent in education:

    map = {
        'Children':	0,
        'Less than 1st grade': 0.5,
        '1st 2nd 3rd or 4th grade':	2.5,
        '5th or 6th grade':	5.5,
        '7th and 8th grade': 7.5,
        '9th grade': 9,
        '10th grade': 10,
        '11th grade': 11,
        'High school graduate': 12,
        'Some college but no degree': 14,
        'Bachelors degree(BA AB BS)': 16,
        'Masters degree(MA MS MEng MEd MSW MBA)': 18,
        'Prof school degree (MD DDS DVM LLB JD)': 20,
        'Doctorate degree(PhD EdD)': 21,
    }
    """
    return series.replace(mapping)

def create_transformer(numcols:List, catcols:List) -> ColumnTransformer:
    """Template ColumnTransformer creation for reference,
    rather than a method, really."""
    transformer = ColumnTransformer(transformers=[
        ('num', RobustScaler(), numcols),
        ('ohe', OneHotEncoder(sparse_output=False), catcols),
        # ('ord_a', OrdinalEncoder(categories=ord_a_list), ordcol_a),
    ])
    dfohe = transformer.fit_transform(df)
    dfohe = pd.DataFrame(dfohe,
                         columns=transformer.get_feature_names_out())

    return transformer


def create_pipeline(transformer, classifier) -> Pipeline:
    pipe = Pipeline([
        ('tfm', transformer)
        ('cls', classifier)
    ])
    return pipe
