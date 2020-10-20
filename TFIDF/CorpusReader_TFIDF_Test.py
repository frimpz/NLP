# Author Frimpong Boadu
# 02/12/2019
# Corpus Reader Class loader

import CorpusReader_TFIDF as Corpus_Reader_Object
from nltk.corpus import state_union,shakespeare,brown


# All constructors are called with default values
#tf="raw", idf="base", stopwords="english", stemmer=PorterStemmer(), ignorecase=True

def cos_print(x, i):
    try:
        print("sim(", x, ",", i, ") :", round(pi.cosine_sim([x, i]), 3))
    except TypeError:
        print("Undefined cosine similarity; division by zero")


print("Corpus : ", "Brown")
pi = Corpus_Reader_Object.CorpusReader_TFIDF(brown)
print(pi.tf_idf_dim()[:15])
for i in pi.tf_idfs():
    print(i, list(pi.tf_idfs()[i].values())[:15])
t = []
for i in pi.tf_idfs():
    t.append(i)
    for x in t:
        cos_print(x, i)

print("")
print(" ****--------------------------------- ****")
print("")
print("Corpus : ", "Shakespeare Text")

pi = Corpus_Reader_Object.CorpusReader_TFIDF(shakespeare)
print(pi.tf_idf_dim()[:15])
for i in pi.tf_idfs():
    print(i, list(pi.tf_idfs()[i].values())[:15])
t = []
for i in pi.tf_idfs():
    t.append(i)
    for x in t:
        cos_print(x, i)

print("")
print(" ****---------------------------------****  ")
print("")


print("Corpus : ", "State of the Union")
pi = Corpus_Reader_Object.CorpusReader_TFIDF(state_union)
print(pi.tf_idf_dim()[:15])
for i in pi.tf_idfs():
    print(i, list(pi.tf_idfs()[i].values())[:15])
t = []
for i in pi.tf_idfs():
    t.append(i)
    for x in t:
        cos_print(x, i)


