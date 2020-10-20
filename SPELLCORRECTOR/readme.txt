# Author Frimpong Boadu
# Author Rezar Rahmsn
# 02/12/2019
# Spell Corrector Class

***** readme for Spellchecker *******

We have a SpellCorrector Class with the following functions:


#Check(x)
it takes a string (consider as a paragraph/sentence) as a parameter. 
It will return a list of tuples. 
Each tuple is made up of the following (p, q, r)

p is the position of the word that is misspelled 
q denotes the actual word that is misspelled
q denotes the actual word that is misspelled


#createkey2(x)
In training our corpus we create a key using the previous n words specified,
We create a similar key with previous n words to fetch suggested words

#getSuggestions(probs, previous)
get suggestions using previous n words

#calculateEditDistance(suggestions, wrong_word)
fine tune sugestions by using edit distance

#calculateProbability(file,totals,freq)
This method is used to calculate probabilities               

#ngrams(file,corpus, size)
We use this function to calculate ngrams

#sentences_from_corpus
we use this method to take prepare


'''
Our suggestions are ordered based on
1. The edit distance between the mispelled word and the word in ascending,e[1][0]
2. The number of previous words used to get the suggestion in descending,-e[1][1]
3. The probability of the suggested word occuring in descending,-e[1][2]

'''

We used brown, reuters and guttenberg for training
gutenberg.txt; brown.txt,reuters.txt

Uncomment these lines from sentences_from_corpus to train:
ngrams("brown.txt",sentences_from_corpus(),ngram=3)
ngrams("guttenberg.txt",sentences_from_corpus(),ngram=3)
ngrams("reuters.txt",sentences_from_corpus(),ngram=3)