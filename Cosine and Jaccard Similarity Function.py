#Fahad Kabir

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
import string

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#dataset 
ndata = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

#Preprocessing 
def clean_text(text):

    #Tokenization
    tokens = word_tokenize(text.lower())

    #Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]

    #Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

#Vectorize 
tfidf_vectorizer = TfidfVectorizer(preprocessor=clean_text)
tfidf_matrix = tfidf_vectorizer.fit_transform(ndata.data)

#cosine similarity function and convert documents to tfidf
def calculate_similarity(doc1, doc2):
   
    doc1_tfidf = tfidf_vectorizer.transform([doc1])
    doc2_tfidf = tfidf_vectorizer.transform([doc2])
    similarity = cosine_similarity(doc1_tfidf, doc2_tfidf)

    return similarity[0][0]

#jaccard similarity function
def calculate_jaccard_similarity(doc1, doc2):

    firstset = set(word_tokenize(doc1.lower()))
    secondset = set(word_tokenize(doc2.lower()))
    intersection = len(firstset.intersection(secondset))
    union = len(firstset.union(secondset))
    similarity = intersection / union
    return similarity

#input document index 
document1_index = int(input("Enter the index of the first document: "))
document2_index = int(input("Enter the index of the second document: "))
document1 = ndata.data[document1_index]
document2 = ndata.data[document2_index]

#returning the result of both functions
cosine_similarity_score = calculate_similarity(document1, document2)
jaccard_similarity_score = calculate_jaccard_similarity(document1, document2)
print("Cosine Similarity between Document {} and Document {}: {:.4f}".format(document1_index, document2_index, cosine_similarity_score))
print("Jaccard Similarity between Document {} and Document {}: {:.4f}".format(document1_index, document2_index, jaccard_similarity_score))
