import time
import pandas as pd
from nltk.tokenize import TweetTokenizer
import numpy as np
from nltk.corpus import stopwords

PATH = "trainTwitter.csv"
SLICE_LEN = 3
REGEX = r'[^a-zA-Z0-9@#!,\'\s]+|X{2,}'
COLUMN = 'tweet'
SPECIAL_CHARS = ('@', '#', '!')
AMOUNT_OF_TASKS = 6
TASK1 = 0
TASK2 = 1
TASK3 = 2
TASK4 = 3
TASK5 = 4  # count upper-case words
TASK6 = 5  # count special chars


def read_file(path):
    file_csv = pd.read_csv(PATH, encoding='latin-1')
    return file_csv


def create_copy_of_file(df):
    return df.copy(deep=True)


def slice_the_data_frame(df):
    return df.iloc[:, 0:SLICE_LEN]


def clean_unrecognized_chars(df):
    df[COLUMN] = df.tweet.str.replace(REGEX, '')
    return df


def get_tokenized_data_with_nltk_tokenizer(df):
    return np.array(df[COLUMN].apply(lambda x: TweetTokenizer().tokenize(x)))


def creat_numpy_array_in_given_size(size_rows, size_cols=AMOUNT_OF_TASKS):
    return np.zeros((size_cols, size_rows), dtype=np.uint8)


def count_stop_words(df):
    stop_words = set(stopwords.words('english'))
    return df[COLUMN].str.split().apply(lambda x: len(set(x) & stop_words))


def use_tokenized_data_and_perform_operations(tokenized_arr):
    new_columns = creat_numpy_array_in_given_size(len(tokenized_arr))
    for index in range(len(tokenized_arr)):
        count_word_amount(tokenized_arr, new_columns, index, TASK1)
        count_amount_of_letters_in_sentence(tokenized_arr, new_columns, index, TASK2)
        count_avg_size_of_words(tokenized_arr, new_columns, index, TASK3)
        count_numeric_chars(tokenized_arr, new_columns, index, TASK4)
        count_upper_case_words(tokenized_arr, new_columns, index, TASK5)
        count_special_chars(tokenized_arr, new_columns, index, TASK6)

    return new_columns


def count_special_chars(tokenized_arr, new_columns, index, task):
    new_columns[task][index] = len([x for x in tokenized_arr[index] if x in SPECIAL_CHARS])


def count_upper_case_words(tokenized_arr, new_columns, index, task):
    new_columns[task][index] = len([x for x in tokenized_arr[index] if x.isupper()])


def count_amount_of_letters_in_sentence(tokenized_arr, new_columns, index, task):
    new_columns[task][index] = len(' '.join(tokenized_arr[index]))


def count_word_amount(tokenized_arr, new_columns, index, task):
    new_columns[task][index] = len(tokenized_arr[index])


def count_avg_size_of_words(tokenized_arr, new_columns, index, task):
    try:
        new_columns[task][index] = len(''.join(tokenized_arr[index])) / len(tokenized_arr[index])
    except ZeroDivisionError as e:
        new_columns[task][index] = 0


def count_numeric_chars(tokenized_arr, new_columns, index, task):
    new_columns[task][index] = len([x for x in tokenized_arr[index] if x.isdigit()])


def apply_functions_over_data_frame(data_frame, tokenized_arr):
    new_cols = use_tokenized_data_and_perform_operations(tokenized_arr)
    data_frame['word_count'] = new_cols[TASK1]
    data_frame['letters_count'] = new_cols[TASK2]
    data_frame['avg_letter_count'] = new_cols[TASK3]
    data_frame['stop_words'] = count_stop_words(data_frame)
    data_frame['numeric_chars'] = new_cols[TASK4]
    data_frame['upper_cases_words'] = new_cols[TASK5]
    data_frame['special_chars'] = new_cols[TASK6]
    return data_frame


def main():
    print("start")
    start_time = time.time()
    file = read_file(PATH)
    file_copy = create_copy_of_file(file)
    sliced_file_without_empty_cols = slice_the_data_frame(file_copy)
    cleaned_file_from_unrecognized_chars = clean_unrecognized_chars(sliced_file_without_empty_cols)
    tokenized_data = get_tokenized_data_with_nltk_tokenizer(cleaned_file_from_unrecognized_chars)
    df = apply_functions_over_data_frame(cleaned_file_from_unrecognized_chars, tokenized_data)
    print("--- %s seconds ---" % (time.time() - start_time))


main()
