import pytest
from main_file import *
from functools import reduce

def test_file_opened_correctly():
    file_opener = read_file(PATH)
    assert (file_opener is not None)


def test_copy_is_created():
    assert create_copy_of_file(file) is not None


def test_data_frame_is_sliced_correctly():
    assert len(slice_the_data_frame(file)) == SLICE_LEN + 1


def test_clean_unrecognized_chars():
    not_contain_unrecognized_chars = [(clean_unrecognized_chars(file)).tweet.str.contains(r'^[ A-Za-z0-9_@./#!]', regex=True)]
    is_all_true = reduce(lambda x,y : x and y, not_contain_unrecognized_chars)
    assert is_all_true[1]
