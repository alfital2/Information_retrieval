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
    is_all_false = reduce(lambda x, y: x and y, not_contain_unrecognized_chars)
    assert not is_all_false[1]


def test_count_amount_of_words():
    len_of_first_tweet_is_18 = len(TweetTokenizer().tokenize(file['tweet'][0]))
    assert len_of_first_tweet_is_18 == 18



