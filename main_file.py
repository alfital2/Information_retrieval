import pandas as pd
from nltk.tokenize import TweetTokenizer
import numpy as np
from nltk.corpus import stopwords

PATH = "trainTwitter.csv"
SLICE_LEN = 3
REGEX = r'[^a-zA-Z0-9@#!,\'\s]+|X{2,}'
COLUMN = 'tweet'


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


def count_word_amount(tokenized_data):
    new_column = creat_numpy_array_in_given_size(len(tokenized_data))

    for index in range(len(tokenized_data)):
        new_column[index] = len(tokenized_data[index])

    return new_column


def get_tokenized_data_with_nltk_tokenizer(df):
    return np.array(df[COLUMN].apply(lambda x: TweetTokenizer().tokenize(x)))


def creat_numpy_array_in_given_size(size):
    return np.arange(size, dtype=np.uint8)


def count_amount_of_letters_in_sentence(tokenized_data):
    new_column = creat_numpy_array_in_given_size(len(tokenized_data))

    for index in range(len(tokenized_data)):
        new_column[index] = len(' '.join(tokenized_data[index]))

    return new_column


def count_avg_size_of_words(tokenized_arr):
    new_column = creat_numpy_array_in_given_size(len(tokenized_arr))

    for index in range(len(tokenized_arr)):
        try:
            new_column[index] = len(''.join(tokenized_arr[index])) / len(tokenized_arr[index])
        except ZeroDivisionError as e:
            new_column[index] = 0
    return new_column


def count_stop_words(df):
    stop_words = set(stopwords.words('english'))
    return df['tweet'].str.split().apply(lambda x: len(set(x) & stop_words))


def count_numeric_chars(tokenized_arr):
    new_column = creat_numpy_array_in_given_size(len(tokenized_arr))

    for index in range(len(tokenized_arr)):
        new_column[index] = len([x for x in tokenized_arr[index] if x.isdigit()])
        if new_column[index]!=0:
            print(index)

    return new_column

def apply_functions_over_data_frame(data_frame, tokenized_arr):
    data_frame['word_count'] = count_word_amount(tokenized_arr)
    data_frame['letters_count'] = count_amount_of_letters_in_sentence(tokenized_arr)
    data_frame['avg_letter_count'] = count_avg_size_of_words(tokenized_arr)
    data_frame['stop_words'] = count_stop_words(data_frame)
    data_frame['numeric_chars'] = count_numeric_chars(tokenized_arr)
    return data_frame


file = read_file(PATH)
file_copy = create_copy_of_file(file)
sliced_file_without_empty_cols = slice_the_data_frame(file_copy)
cleaned_file_from_unrecognized_chars = clean_unrecognized_chars(sliced_file_without_empty_cols)
tokenized_data = get_tokenized_data_with_nltk_tokenizer(cleaned_file_from_unrecognized_chars)
df = apply_functions_over_data_frame(cleaned_file_from_unrecognized_chars, tokenized_data)

print(df)
