import pytest
from main_file import *


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5


def test_other_func():
    assert funcTest() == 0.5
