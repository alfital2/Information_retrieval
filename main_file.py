import time
import pandas as pd
from nltk import LancasterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np
from nltk.corpus import stopwords
from textblob import Word
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import squarify
import copy

PATH = "trainTwitter.csv"
SLICE_LEN = 4
REGEX = r'[^a-zA-Z0-9@#!,\'\s]+|X{2,}|[,]|[!]'
COLUMN = 'tweet'
SPECIAL_CHARS = ('@', '#', '!')
AMOUNT_OF_TASKS = 6
TASK1 = 0
TASK2 = 1
TASK3 = 2
TASK4 = 3
TASK5 = 4  # count upper-case words
TASK6 = 5  # count special chars
WORD_COUNT = 'word_count'
LETTERS_COUNT = 'letters_count'
AVG_LETTERS_COUNT = 'avg_letter_count'
STOP_WORDS = 'stop_words'
NUMERIC_CHARS = 'numeric_chars'
UPPER_CASE_WORDS = 'upper_cases_words'
SPECIAL_CHARS = 'special_chars'


def read_file(path=PATH):
    file_csv = pd.read_csv(path, encoding='latin-1')
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
    data_frame[WORD_COUNT] = new_cols[TASK1]
    data_frame[LETTERS_COUNT] = new_cols[TASK2]
    data_frame[AVG_LETTERS_COUNT] = new_cols[TASK3]
    data_frame[STOP_WORDS] = count_stop_words(data_frame)
    data_frame[NUMERIC_CHARS] = new_cols[TASK4]
    data_frame[UPPER_CASE_WORDS] = new_cols[TASK5]
    data_frame[SPECIAL_CHARS] = new_cols[TASK6]
    return data_frame


# --------------- VISUALIZATION PART:

def get_all_words_from_tweets_before_preprocess(data):
    return "".join(str(word) + ' ' for tweet in data for word in tweet)


def creat_word_cloud(words):
    word_cloud = WordCloud(background_color="white", collocations=False).generate(words)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_twenty_most_common_words(data, amount=20):
    return Counter(" ".join(data[COLUMN]).split()).most_common(amount)


def creat_tile_for_twenty_most_common(twenty_most_common):
    volume = [x[1] for x in twenty_most_common]
    labels = [x[0] for x in twenty_most_common]
    color_list = ['#0f7216', '#b2790c', '#ffe9a3',
                  '#f9d4d4', '#d35158', '#ea3033']
    squarify.plot(sizes=volume, label=labels, color=color_list, alpha=0.7)
    plt.show()


def create_sorted_series_of_unique_values_counts(dataframe):
    return dataframe.value_counts().sort_index()


def create_arrays_of_x_axis_and_y_axis(values, threshold=0):
    unique_values_count = create_sorted_series_of_unique_values_counts(values)
    amount_of_words, tweets_amount = [], []
    to_remove = unique_values_count[unique_values_count <= threshold].index

    for x, y in unique_values_count.items():
        if x not in to_remove:
            amount_of_words.append(x)
            tweets_amount.append(y)

    return [amount_of_words, tweets_amount]


def create_plot_of_words_occurrence_relative_to_tweets(word_count):
    amount_of_words, tweets_amount = create_arrays_of_x_axis_and_y_axis(word_count, 10)
    plt.xticks(amount_of_words)
    plt.grid()
    plt.xlabel("amount of words in tweet")
    plt.ylabel("amount of tweets")
    plt.scatter(amount_of_words, tweets_amount)
    plt.show()


def create_plot_of_chars_amount_relative_to_tweets(letters_count):
    amount_of_letters, tweets_amount = create_arrays_of_x_axis_and_y_axis(letters_count, 0)
    plt.grid()
    plt.xlabel("amount of letters in tweet")
    plt.ylabel("amount of tweets")
    plt.scatter(amount_of_letters, tweets_amount)
    plt.show()


def create_plot_of_stop_words_amount_relative_to_tweets(stop_words):
    amount_of_stop_words, tweets_amount = create_arrays_of_x_axis_and_y_axis(stop_words)
    plt.grid()
    plt.xlabel("amount of stop words in tweet")
    plt.ylabel("amount of tweets")
    plt.scatter(amount_of_stop_words, tweets_amount)
    plt.show()


def create_plot_of_numeric_digits_amount_relative_to_tweets(numeric_values):
    amount_of_numeric_values, tweets_amount = create_arrays_of_x_axis_and_y_axis(numeric_values, 5)
    plt.grid()
    plt.xlabel("amount of numeric values in tweet")
    plt.ylabel("amount of tweets")
    plt.scatter(amount_of_numeric_values, tweets_amount)
    plt.show()


def get_offensive_tweets_data_frame(dataframe, stop_words, val=1):
    offensive_tweets_dataframe = dataframe[(dataframe["label"] == val)]
    offensive_tweets_dataframe['tweet'] = offensive_tweets_dataframe['tweet'].apply(lambda x: \
                                                                                        ' '.join(
                                                                                            [word for word in x.split()
                                                                                             if
                                                                                             word not in stop_words]))
    return offensive_tweets_dataframe.replace(',', '', regex=True)


def create_histogram_to_show_most_offensive_words(dataframe):
    stop_words = set(stopwords.words('english'))
    stop_words.add('@user')
    offensive_tweets_dataframe = get_offensive_tweets_data_frame(dataframe, stop_words)
    most_common_offensive_words = get_twenty_most_common_words(offensive_tweets_dataframe, 10)
    words = [x[0] for x in most_common_offensive_words]
    frequency = [x[1] for x in most_common_offensive_words]

    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(words, frequency, align='center')
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.show()


def create_plot_to_compare_offensive_tweets_to_no_offensive(dataframe):
    stop_words = set(stopwords.words('english'))
    stop_words.add('@user')
    offensive_tweets_dataframe = get_offensive_tweets_data_frame(dataframe, stop_words)
    regular_dataframe = get_offensive_tweets_data_frame(dataframe, stop_words, 0)
    avg_offensive = (offensive_tweets_dataframe[WORD_COUNT].sum()) / len(offensive_tweets_dataframe)
    avg_regular = (regular_dataframe[WORD_COUNT].sum()) / len(regular_dataframe)
    fig, ax = plt.subplots()
    x = ['offensive', 'non offensive']
    y = [avg_offensive, avg_regular]
    ax.bar(x, y)
    plt.show()


# -------------------------------------- End of visualization


# ------------------------------------------ lab 2
def lemmatize(array, word):
    lemma = Word(word).lemmatize()
    array.remove(word)
    array.append(lemma)


def correct_spelling(array, word):
    correct_word = Word(word).correct()
    if word != correct_word:
        array.remove(word)
        array.append(correct_word)


def stem(array, word):
    stemmed = LancasterStemmer().stem(word)
    array.remove(word)
    array.append(stemmed)


def process_tokens_by_operation(tokenized_data, func):
    copy_tokenized_arr = copy.deepcopy(tokenized_data)

    for array in copy_tokenized_arr:
        for word in list(array):
            func(array, word)

    return copy_tokenized_arr


def clean_word_from_tokens_array(tokenized_data, word_to_remove):
    copy_tokenized_arr = copy.deepcopy(tokenized_data)

    for array in copy_tokenized_arr:
        for word in list(array):
            if word == word_to_remove:
                array.remove(word)

    return copy_tokenized_arr


def main():
    # print("start")
    # start_time = time.time()
    # file = read_file(PATH)
    # file_copy = create_copy_of_file(file)
    # sliced_file_without_empty_cols = slice_the_data_frame(file_copy)
    # cleaned_file_from_unrecognized_chars = clean_unrecognized_chars(sliced_file_without_empty_cols)
    # tokenized_data = get_tokenized_data_with_nltk_tokenizer(cleaned_file_from_unrecognized_chars)
    # df = apply_functions_over_data_frame(cleaned_file_from_unrecognized_chars, tokenized_data)
    # df.to_csv("processed.csv")
    # print("--- %s seconds ---" % (time.time() - start_time))

    # print("visualization")
    # print("start")
    # start_time = time.time()
    # all_the_words_from_tweets_before_preprocess = get_all_words_from_tweets_before_preprocess(tokenized_data)
    # creat_word_cloud(all_the_words_from_tweets_before_preprocess)
    # twenty_most_common_words_in_data_frame = get_twenty_most_common_words(df)
    # creat_tile_for_twenty_most_common(twenty_most_common_words_in_data_frame)
    # create_plot_of_words_occurrence_relative_to_tweets(df[WORD_COUNT])
    # create_plot_of_chars_amount_relative_to_tweets(df[LETTERS_COUNT])
    # create_plot_of_stop_words_amount_relative_to_tweets(df[STOP_WORDS])
    # create_plot_of_numeric_digits_amount_relative_to_tweets(df[NUMERIC_CHARS])
    # create_histogram_to_show_most_offensive_words(df)
    # create_plot_to_compare_offensive_tweets_to_no_offensive(df)
    # print("--- %s seconds ---" % (time.time() - start_time))

    #------------------------------------------------------------------------------------------- lab1 ends here

    # cleaned_tokens_array = clean_word_from_tokens_array(tokenized_data, '@user') # all the token array without 'user'
    # cleaned_tokens_10_tweets   = cleaned_tokens_array[0:10]
    # spell_correct_tokens = process_tokens_by_operation(cleaned_tokens_10_tweets,correct_spelling)

    spell_correct_tokens = [list(['when', 'a', 'fath', 'is', 'dysfunct', 'and', 'is', 'so', 'self', 'he', 'drag', 'his', 'into', 'his', 'dysfunct', 'kiss', 'run']),
     list(['thank', 'for', 'credit', 'i', 'us', 'caus', 'they', 'off', 'wheelchair', 'in', '#getthanked', 'left', 'canst', 'dont', 'van', 'pox', 'disappoint']),
     list(['yo', 'majesty', 'midday']),
     list(['i', 'lov', 'u', 'tak', 'with', 'u', 'al', 'the', 'tim', 'in', '0000', '000', 'model', 'urg']),
     list(['factsguid', 'socy', 'now', 'mot']),
     list(['22', 'hug', 'fan', 'far', 'and', 'big', 'talk', 'bef', 'they', 'leav', 'chao', 'and', 'pay', 'disput', 'when', 'they', 'get', 'ther', '#allshowandnogo']),
     list(['camp', 'tomorrow', 'dandy']),
     list(['the', 'next', 'school', 'year', 'is', 'the', 'year', 'for', '0', 'think', 'about', 'that', '0', 'school', '#actorslife', '#revolutionschool', 'exam', 'canst', 'texa', 'hat', 'imagin', 'girl']),
     list(['we', 'won', 'lov', 'the', 'land', '#clevelandcavaliers', 'fal', 'scar', 'champ', 'cleveland']),
     list(['welcom', 'her', 'so', 'ism', 'it', 'grm'])]
    ### !!!! TO SAVE TIME WE WILL USE THIS ARRAY : !!!

    lemmatized_tokens = process_tokens_by_operation(spell_correct_tokens,lemmatize)
    stemmed_tokens = process_tokens_by_operation(lemmatized_tokens,stem)
    print(stemmed_tokens)


main()
