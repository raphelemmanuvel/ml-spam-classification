# normalization technique
# The linguistic difference with respect to stemming is that lemmatization will enable
# for words which do not have the same root to be grouped together in order for them to
# be processed as one item.


from nltk.stem.wordnet import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokenized_list):
    """Lemmatize verbs in list of tokenized words"""
    lemmas = []
    for tokens in tokenized_list:
        lem_words = [lemmatizer.lemmatize(x,pos='v') for x in (tokens)]
        lemmas.append(lem_words)
    return lemmas
