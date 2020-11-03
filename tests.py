import pytest
from main_file import *
from functools import reduce

def test_file_opened_correctly():
    file_opener = read_file(PATH)
    assert (file_opener is not None)


def test_copy_is_created():
    assert create_copy_of_file(file) is not None


def test_data_frame_is_sliced_correctly():
    assert len(slice_the_data_frame(file).columns) == SLICE_LEN


def test_clean_unrecognized_chars():
    not_contain_unrecognized_chars = [(clean_unrecognized_chars(file)).tweet.str.contains(REGEX, regex=True)]
    print(not_contain_unrecognized_chars)
    is_all_false = reduce(lambda x,y : x and y, not_contain_unrecognized_chars)
    assert is_all_false[1] == False
