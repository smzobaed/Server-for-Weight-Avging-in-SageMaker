# ********************************************
# FUNCTIONS FOR PREPROCESSING
# ********************************************

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import re
import string
import json
import pickle
import yake
import nltk
nltk.download('stopwords')


def get_dtypes():
    '''
    retrieve manual specification of column data types
    makes reading the data faster and lighter
    '''
    dtypes={
        'partnumber': str,
        'descr_part': str,
        'unit_cost': float,
        'unitsales': float,
        'quantity': float,
        'totalcost': float,
        'totalsales': float,
        'laborid': str,
        'hours': float,
        'sales': float,
        'cost': float,
        'complaint': str,
        'cause': str,
        'correction': str,
        'repairorderid': str,
        'opcodeid': str,
        'code': str,
        'descr_opcode': str,
        'clientid': str,
        'vehiclemileagein': float,
        'misccosts': float,
        'greaseoilgas': float,
        'vehicleid': str,
        'vin': str,
        'partnumberkey': str,
        'partid': str,
        'opid': str,
        'systemdescription': str,
        'groupdescription': str,
        'subgroupdescription': str,
        'operationtypedescription': str,
        'literalname': str,
        'mso': str
    }
    return dtypes
    
    
def drop_unlabeled_rows(df):
    '''
    Drops rows for which we have no label
    '''
    df = df.dropna(subset='literalname')
    return df


def select_columns(df, num_cols):
    '''
    Exclude all columns except those explictly listed below
    '''
    df['label'] = df.literalname
    cols = ['label', 'text'] + num_cols
    df = df.loc[:,cols]
    return df


def drop_singletons(df):
    '''
    Drops data for labels with sample size of 1 (train/test split requires N >= 2)
    '''
    label_sample_count = df['label'].value_counts()
    label_sample_count = label_sample_count.to_frame().reset_index().rename(
        columns={'label':'count', 'index':'label'}
    )
    df = pd.merge(df, label_sample_count, on='label', how='left')
    df = df[df['count'] > 1]
    return df


def downsample(df, max_n):
    '''
    randomly samples rows of overrepresented classes up to max_n
    '''
    over_data = df[df['count'] > max_n]
    under_data = df[df['count'] <= max_n]
    groupby_label = over_data.groupby('label', group_keys=False)
    bal_over_data = pd.DataFrame(
        groupby_label.apply(lambda x: x.sample(max_n)).reset_index(drop=True)
    )
    df = pd.concat([bal_over_data, under_data])
    df = df.loc[:, df.columns != 'count']
    return df


def concat(row):
    '''
    Input: row of dynatron data
    Output: string resulting from combining text fields of interest
    '''
    text = f"{row['descr_part']} {row['complaint']} {row['cause']} {row['correction']} {row['descr_opcode']}"
    return text


def concat_text(df):
    '''
    Concatenate text columns
    '''
    df['text'] = df.apply(lambda x: concat(x), axis=1)
    return df


def clean(text):
    '''
    cleans up text string
    '''
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', ' ', text) # remove hyperlinks
    text = re.sub(r'<.*?>', ' ', text) # remove html
    text = re.sub('\n', ' ', text) # remove newlines
    text = re.sub('\t', ' ', text) # remove tabs
    # remove 'customer states' phrases
    text = re.sub('c/s', ' ', text)
    text = re.sub('customer states', ' ', text)
    text = re.sub('customer requests', ' ', text)
    text = re.sub('customer request', ' ', text)
    text = re.sub('nan', ' ', text) # convert missing data to word
    text = re.sub(r'[^\w\s]', ' ', text) # removes punctuation
    text = re.sub(r'[0-9]', ' ', text) # removes digits
    text = re.sub('\s\s+', ' ', text) # remove extra spaces
    return text
    
    
def clean_text(df):
    '''
    Cleans up strings in the text column
    '''
    df['text'] = df['text'].apply(lambda x: clean(x))
    return df


def extract(text):
    '''
    extracts keywords from text
    '''
    extractor = yake.KeywordExtractor(n=1, dedupLim=.9, dedupFunc='seqm', windowsSize=1, top=5)
    keywords = extractor.extract_keywords(text)
    word_list = []
    for kw in keywords:
        word_list += [kw[0]]
    return ' '.join(word_list)


def extract_keywords(df):
    '''
    Extracts keywords from text column
    '''
    keyword_column = df['text'].apply(lambda x: extract(x))
    return keyword_column


def vectorize_text(df, key_text):
    '''
    Applies TFIDF vectorizer to keywords, replaces text with resulting numeric features
    '''
    vectorizer = TfidfVectorizer(stop_words=set(nltk.corpus.stopwords.words('english')))
    vec_text = vectorizer.fit_transform(key_text)
    vec_text_df = pd.DataFrame(vec_text.toarray())
    df = df.loc[:, df.columns != 'text'].reset_index(drop=True)
    df = pd.concat([df, vec_text_df], axis=1)
    return df, vectorizer


def split(df, test_size):
    '''
    Performs a y-stratified, shuffled train/test split on the data
    '''
    X = df.loc[:, df.columns != 'label']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=0,
        stratify=y,
        shuffle=True
    )
    return X_train, X_test, y_train, y_test    


def encode_labels(y_train, y_test):
    '''
    converts string labels to integers
    '''
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    return y_train, y_test, encoder


def zscore(X_train, X_test, num_cols):
    '''
    Mean-center and sd-scale
    Fit to training data, and applied to training and test data
    '''
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, scaler


def reassemble(X, y):
    '''
    creates a dataframe from the preprocessed features (X) and targets (y)
    y is now the first column
    '''
    new_df = pd.DataFrame(X)
    new_df['label'] = y
    first_col = new_df.pop('label')
    new_df.insert(0, 'label', first_col)
    return new_df    
    

def save_prep_data(train_df, test_df):
    '''
    saves preprocessed data without headers or indices
    '''
    path = '/opt/ml/processing'
    train_df.to_csv(f'{path}/train/data.csv', header=False, index=False)
    test_df.to_csv(f'{path}/test/data.csv', header=False, index=False)
    
    
def save_prep_models(vectorizer, label_encoder, scaler):
    '''
    saves preprocessing models as pickle files
    '''
    path = '/opt/ml/processing/model/'
    if not os.path.exists(path):
        # Create a new directory because it does not exist 
        os.makedirs(path)
        print(f'Added path {path}')
        
    with open(f'{path}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{path}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(f'{path}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)