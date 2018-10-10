import re
import string
from collections import Counter
import numpy as np

def preprocess_ST_message(text):
    """
    Preprocesses raw message data for analysis
    :param text: String. ST Message
    :return: List of Strings.  List of processed text tokes
    """
    # Define ST Regex Patters
    REGEX_PRICE_SIGN = re.compile(r'\$(?!\d*\.?\d+%)\d*\.?\d+|(?!\d*\.?\d+%)\d*\.?\d+\$')
    REGEX_PRICE_NOSIGN = re.compile(r'(?!\d*\.?\d+%)(?!\d*\.?\d+k)\d*\.?\d+')
    REGEX_TICKER = re.compile('\$[a-zA-Z]+')
    REGEX_USER = re.compile('\@\w+')
    REGEX_LINK = re.compile('https?:\/\/[^\s]+')
    REGEX_HTML_ENTITY = re.compile('\&\w+')
    REGEX_NON_ACSII = re.compile('[^\x00-\x7f]')
    REGEX_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation.replace('<', '')).replace('>', ''))
    REGEX_NUMBER = re.compile(r'[-+]?[0-9]+')

    text = text.lower()

    # Replace ST "entitites" with a unique token
    text = re.sub(REGEX_TICKER, ' <TICKER> ', text)
    text = re.sub(REGEX_USER, ' <USER> ', text)
    text = re.sub(REGEX_LINK, ' <LINK> ', text)
    text = re.sub(REGEX_PRICE_SIGN, ' <PRICE> ', text)
    text = re.sub(REGEX_PRICE_NOSIGN, ' <NUMBER> ', text)
    text = re.sub(REGEX_NUMBER, ' <NUMBER> ', text)
    # Remove extraneous text data
    text = re.sub(REGEX_HTML_ENTITY, "", text)
    text = re.sub(REGEX_NON_ACSII, "", text)
    text = re.sub(REGEX_PUNCTUATION, "", text)
    # Tokenize and remove < and > that are not in special tokens
    words = " ".join(token.replace("<", "").replace(">", "")
                     if token not in ['<TICKER>', '<USER>', '<LINK>', '<PRICE>', '<NUMBER>']
                     else token
                     for token
                     in text.split())

    return words

"""
To work with raw text we need some encoding from words to numbers for our algorithm to work with the inputs. The first step of doing this is keeping a collection of our full vocabularly and creating a mapping of each word to a unique index. We will use this word to index mapping in a little bit to prep out messages for analysis.

Note that in practice we may want to only include the vocabularly from our training set here to account for the fact that we will likely see new words when our model is out in the wild when we are assessing the results on our validation and test sets. Here, for simplicity and demonstration purposes, we will use our entire data set.
"""
def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict maps a vocab word to and integeter
             The second maps an integer back to to the vocab word
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

"""
remove zero length messages
"""
def drop_empty_messages(messages, labels):
    """
    Drop messages that are left empty after preprocessing
    :param messages: list of encoded messages
    :return: tuple of arrays. First array is non-empty messages, second array is non-empty labels
    """
    non_zero_idx = [ii for ii, message in enumerate(messages) if len(message) != 0]
    messages_non_zero = np.array([messages[ii] for ii in non_zero_idx])
    labels_non_zero = np.array([labels[ii] for ii in non_zero_idx])
    return messages_non_zero, labels_non_zero

"""
The last thing we need to do is make our message inputs the same length. In our case, the longest message is 244 words. LSTMs can usually handle sequence inputs up to 500 items in length so we won't truncate any of the messages here. We need to Zero Pad the rest of the messages that are shorter. We will use a left padding that will pad all of the messages that are shorter than 244 words with 0s at the beginning. So our encoded "I am bullish" messages goes from [1, 234, 5345] (length 3) to [0, 0, 0, 0, 0, 0, ... , 0, 0, 1, 234, 5345] (length 244).
"""
def zero_pad_messages(messages, seq_len):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param seq_ken: Input int, maximum sequence input length
    :return: numpy array.  The encoded labels
    """
    messages_padded = np.zeros((len(messages), seq_len), dtype=int)
    for i, row in enumerate(messages):
        messages_padded[i, -len(row):] = np.array(row)[:seq_len]

    return np.array(messages_padded)

def get_batches(x, y, batch_size=100):
    """
    Batch Generator for Training
    :param x: Input array of x data
    :param y: Input array of y data
    :param batch_size: Input int, size of batch
    :return: generator that returns a tuple of our x batch and y batch
    """
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
