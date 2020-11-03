import pandas as pd
from nltk.tokenize import TweetTokenizer

PATH = "trainTwitter.csv"
SLICE_LEN = 3
REGEX = r'[^a-zA-Z@#!,\'\s]+|X{2,}'


def read_file(path):
    file_csv = pd.read_csv(PATH, encoding='latin-1')
    return file_csv


def create_copy_of_file(df):
    return df.copy(deep=True)


def slice_the_data_frame(df):
    return df.iloc[:, 0:SLICE_LEN]


def clean_unrecognized_chars(df):
    df['tweet'] = df.tweet.str.replace(REGEX, '')
    return df


def count_word_amount(df):
    return df.tweet.apply(lambda x: len(TweetTokenizer().tokenize(x)))




def apply_functions_over_data_frame(df):
    df['word_count'] = count_word_amount(df)
    print(df)



file = read_file(PATH)
file_copy = create_copy_of_file(file)
sliced_file_without_empty_cols = slice_the_data_frame(file_copy)
cleaned_file_from_unrecognized_chars = clean_unrecognized_chars(sliced_file_without_empty_cols)
#file_with_count_column = count_word_amount(cleaned_file_from_unrecognized_chars)
apply_functions_over_data_frame(cleaned_file_from_unrecognized_chars)
