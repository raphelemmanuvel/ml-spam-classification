# Stemming is used to normalize parts of text data.

# When you are using a verb which is conjugated in multiple tenses
# throughout a document you would like to process, stemming will shorten
# all of these conjugated verbs to the shortest length of characters possible;
# it will preserve the root of the verb in this case.

# Stemming is done for all types of words, adjectives and more


#NLTK provides several stemmer interfaces like Porter stemmer, #Lancaster Stemmer, Snowball Stemmer

from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


available_stemmers = {
  'Porter': PorterStemmer(),
  'Snowball' : SnowballStemmer(language = 'english'),
  'Lancaster': LancasterStemmer()
}

# Arguments :  tokens_list -> list of tokenized preprocessed words
#              stemmer -> Name of stemmer to be used (eg:Porter, Snowball)

def stemmize_tokens(tokens_list,stemmer):
    stemmed_list = []
    for tokens in tokens_list:
        stems = []
        for t in tokens:
            stems.append(available_stemmers[stemmer].stem(t))
        stemmed_list.append(stems)
    return stemmed_list
