import pandas as pd

PATH = "trainTwitter.csv"
SLICE_LEN = 4


def read_file(path):
    file_csv = pd.read_csv(PATH, encoding='latin-1')
    return file_csv


def create_copy_of_file(file):
    return file.copy(deep=True)


def slice_the_data_frame(file):
    return file.iloc[:,0:SLICE_LEN].head()


def clean_unrecognized_chars(file):
    return file.tweet.str.replace(r'[^a-zA-Z\s]+|X{2,}', '')


file = read_file(PATH)
file_copy = create_copy_of_file(file)
sliced_file_without_empty_cols = slice_the_data_frame(file_copy)
print(sliced_file_without_empty_cols)
print(clean_unrecognized_chars(sliced_file_without_empty_cols))