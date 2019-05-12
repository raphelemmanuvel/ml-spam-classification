import warnings
warnings.filterwarnings("ignore")
import re, string, unicodedata # Regular expressions
import contractions            # Library, solely for expanding contractions
import inflect                 # Correctly generate plurals, singular nouns, ordinals, indefinite articles; convert numbers to words

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords   #Stopwords corpus
from nltk.tokenize import word_tokenize,sent_tokenize #Word Tokenizer

from bs4 import BeautifulSoup  # Extract data from HTML and XML documents

def preprocess(text):
    text = strip_html(text)
    text = remove_url_params(text)
    text = remove_punctuations(text)
    text = remove_special_line_chars(text)
    text = remove_numbers(text)
    text = re.sub(r'(lt.*gt)','',text)
    text = strip_white_spaces(text)
    text = lower_case_text(text)
    text = replace_contractions(text)
    clean_data = text
    return text

def get_tokenized_text(text):
    tokenized_text = tokenize_text(text)
    words = [word for word in tokenized_text if word not in stopwords()] # remove stopwords from the tokenized words list
    words = remove_non_ascii(words)
    meaningful_words = [w for w in words if len(w)>2] # remove words that doesn't convey any meaning like as,us etc based on their number of charcaters
    return meaningful_words



def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def stopwords():
    return nltk.corpus.stopwords.words('english')

def remove_special_line_chars(text):
    """Replace all special characters in a text such as \t \n \s"""
    regex = re.compile(r'[\n\r\t]')
    return regex.sub(" ", text)

def remove_numbers(text):
    """Replace all numbers in a text"""
    return re.sub(r'\d+','',text)

def remove_punctuations(text):
    """Replace all punctuations in a text"""
    return re.sub(r'[^\w\s]', '', text)

def remove_url_params(text):
     """Replace all url paramas such as www,http,https in a text"""
     text = text.replace("http", " ")
     text = text.replace("www", " ")
     text = text.replace("http", " ")
     text = text.replace("https", " ")
     text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
     return text

def tokenize_text(sentence):
    """Tokenize a sentence to word tokens"""
    return nltk.word_tokenize(sentence)

def strip_html(text):
    """Strips html tags ion a text"""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def lower_case_text(text):
    """Converts text to lowercase"""
    return text.lower()

def strip_white_spaces(text):
    """Strips leading and tariling white spaces in a text along with duplicated spaces"""
    text = re.sub(' +', ' ',text)
    text = text.strip()
    return text
