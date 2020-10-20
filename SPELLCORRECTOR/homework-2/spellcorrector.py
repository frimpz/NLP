# Author Frimpong Boadu
# Author Rezar Rahmsn
# 02/12/2019
# Spell Corrector Class


from nltk.corpus import words,wordnet,stopwords
from nltk import word_tokenize
from nltk import edit_distance
from json import load
from string import punctuation


class SpellCorrector:

    # Constructor setting up initialization
    def __init__(self, ):
        self.stopwords = set(stopwords.words('english'))
        #self.wordlist = words.words()
        #training files
        self.files = ['brown.txt', 'gutenberg.txt', 'reuters.txt']

    # check function
    def check(self, x):
        sol = []
        try:
            tokens = word_tokenize(x)
            for pos,word in enumerate(tokens):
                if word not in punctuation:
                    if self.stopwords is not None and word not in self.stopwords:
                        #if word not in self.wordlist:
                        if not wordnet.synsets(word):
                            self.mispelled = (pos,word)
                            prev = self.getprevious(3,pos,tokens)

                            suggestions = {}
                            for file in self.files:
                                data = self.openTrained(file)
                                temp_suggestions = self.getSuggestions(data, prev)
                                temp_suggestions = self.calculateEditDistance(temp_suggestions, word)

                                for key in temp_suggestions.keys():
                                    if key in suggestions.keys():
                                        if suggestions[key][0] > temp_suggestions[key][0]:
                                            suggestions[key] = temp_suggestions[key]
                                    else:
                                        suggestions[key] = temp_suggestions[key]

                            '''
                            This method is used to order our suggestions on 3 values
                            1. The edit distance between the mispelled word and the word in ascending,e[1][0]
                            2. The number of previous words used to get the suggestion in descending,-e[1][1]
                            3. The probability of the suggested word occuring in descending,-e[1][2]
                            *** We tried it on 6 different combinations and we chose this as our final ***
                            *** The other combinations are commented under the code
                            '''
                            final = sorted(suggestions.items(), key=lambda e: (e[1][0], -e[1][1], -e[1][2]))[0:5]

                            temp = []
                            for i in final:
                                temp.append(i[0])
                            sol.append((pos,word,temp))
            return(sol)

        except TypeError:
            print("Check function accepts only strings")

    #get previous n words from mispelled words
    def getprevious(self,ngram,position,words):
        prev = []
        for i in range(position-ngram,position):
            try:
                if i >= 0:
                    prev.append(words[i])
            except IndexError:
                pass
        if not prev:
            prev = [""]
        return prev

    #open trained
    def openTrained(self,file):
        with open(file) as json_file:
            return load(json_file)

    #In training our corpus we create a key using the previous n words specified,
    # We create a similar key with previous n words to fetch suggested words
    def createkey2(self,x):
        key = ""
        for i in x:
            if key == "":
                key = i.lower().strip()
            else:
                key = key+"-"+i.lower().strip()
        if key == "":
            key = "__"
        return key

    #get suggestions using previous n words
    def getSuggestions(self, probs, previous):
        keys = []
        lent = len(previous)
        rang = lent-1
        for i in range(lent):
            keys.append(self.createkey2(previous[rang-i:]))
        probabilities = {}
        for pos,key in enumerate(keys):
            if key in probs.keys():
                probabilities = {**probabilities,**{pos:probs[key]}}
        return probabilities

    # get fine sugestions by using edit distance
    def calculateEditDistance(self,suggestions,wrong_word):
        sug = {}
        for keys in suggestions:
            for key in suggestions[keys].keys():
                sug[key] = (edit_distance(key, wrong_word),keys,suggestions[keys][key])
        return sug






'''
These methods are used to order our suggestions on 3 values
1. The edit distance between the mispelled word and the word in ascending,e[1][0]
2. The number of previous words used to get the suggestion in descending,-e[1][1]
3. The probability of the suggested word occuring in descending,-e[1][2]

final = sorted(suggestions.items(), key=lambda e: (-e[1][1], e[1][0],-e[1][2]))[0:5]
print("prio edit proba "+ str(final))

final = sorted(suggestions.items(), key=lambda e: (-e[1][1],-e[1][2], e[1][0]))[0:5]
print("prio proba edit "+ str(final))

****** chosen one ***** 
final = sorted(suggestions.items(), key=lambda e: (e[1][0], -e[1][1],-e[1][2]))[0:5]
print("edit prio proba "+ str(final))

final = sorted(suggestions.items(), key=lambda e: (e[1][0], -e[1][2],-e[1][1]))[0:5]
print("edit proba prio "+ str(final))

final = sorted(suggestions.items(), key=lambda e: (-e[1][2], -e[1][1],e[1][0]))[0:5]
print("proba prio edit "+ str(final))

final = sorted(suggestions.items(), key=lambda e: (-e[1][2], e[1][0],-e[1][1]))[0:5]
print("proba edit prio "+ str(final))
'''