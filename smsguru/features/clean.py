from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

stop = set(stopwords.words("german"))
tokenizer= RegexpTokenizer(r"\w+")

def clean_language(sen):
    tokens = tokenizer.tokenize(sen)
    return [w for w in tokens if w not in stop]