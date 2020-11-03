import pytest
from main_file import *


def test_file_opened_correctly():
    file_opener = read_file(PATH)
    assert (file_opener is not None)


def test_copy_is_created():
    assert create_copy_of_file(file) is not None


def test_data_frame_is_sliced_correctly():
    assert len(slice_the_data_frame(file)) == SLICE_LEN + 1
