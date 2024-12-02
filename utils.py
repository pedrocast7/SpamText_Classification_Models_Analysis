
# Importing NLTK for natural language processing
import nltk

# Downloading NLTK data
nltk.download('stopwords')   # to deal with stopwords data
nltk.download('punkt')       # to deal with tokenizer data


from nltk.corpus import stopwords    # to process stopwords


# Porter Stemmer for text stemming (reduce complexity of vocabulary)
from nltk.stem.porter import PorterStemmer

# to handle special characters
import string



def transform_text(text) -> list:
    '''Function that process the text received by putting to lowercase, tokenizing,
      removing special characters/stop words/punctuation'''

    # instantiating the Porter Stemmer
    por_stem = PorterStemmer()
    
    # text to lowercase
    text = text.lower()
    
    # tokenization with NLTK
    text = nltk.word_tokenize(text)
    
    # special characters removal
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # stop words and punctuation removal
    text = y[:]
    y.clear()
    
    # loop through the tokens and removing stopwords/punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    # stemming
    text = y[:]
    y.clear() #empty list
    for i in text:
        y.append(por_stem.stem(i))
    
    # joining the processed tokens back to one single string
    return " ".join(y)