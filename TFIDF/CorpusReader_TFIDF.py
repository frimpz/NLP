# Author Frimpong Boadu
# 02/12/2019
# Corpus Reader Class


from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as stpwds
from math import log2
from math import sqrt


class CorpusReader_TFIDF:

    # constructor
    def __init__(self, corpus, tf="raw", idf="base", stopwords="english", stemmer=PorterStemmer(), ignorecase=True):
        self.corpus = corpus
        self.tf = tf
        self.idf = idf
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.ignorecase = ignorecase

        doc = {}
        for i in self.fileids():
            doc[i] = self.words(i)

        #Calculations for tf,idf and weihts are sone in constructor; the methods return the values.
        # Calculations for tf,idf and weights are done in constructor;
        # the methods return the values.

        doc = self.__apply_considerations(doc)

        number_of_docx = len(doc)

        # finding how many times each word appear in a document
        # create a unique idf for each word
        # at this point idf contains keys and values; values is the number of documents that contain the key
        self.doc_idf = {}
        temp = {}
        self.weights = {}
        for fileid in doc:
            doc[fileid] = dict(self.__term_frequency(doc[fileid], self.tf))
            tfidf = {}
            for word in doc[fileid]:
                if word not in self.doc_idf:
                    self.doc_idf[word] = self.__inverse_document_frequency(1,number_of_docx,self.idf)
                    temp[word] = 1
                else:
                    self.doc_idf[word] = self.__inverse_document_frequency(temp[word] + 1,number_of_docx,self.idf)
                    temp[word] = temp[word] + 1
                tfidf[word] = doc[fileid][word] * self.doc_idf[word]
            self.weights[fileid] = tfidf

        self.weights = self.__calculate_weights(doc)

# This block of code makes each vector size consistent by appending values in one list which are not in the other
#  All list are then sorted to make them equivalent
        for keys in self.doc_idf:
            for pi,files in list(self.weights.items()):
                if keys not in files:
                    files[keys] = 0
                self.weights[pi] = files
        for i in self.weights:
            self.weights[i] =(dict(sorted(self.weights[i].items())))


    def test(self):
        return self.corpus.fileids()

    def fileids(self):
        return self.corpus.fileids()

    def raw(self, fileids=None):
        return self.corpus.raw(fileids)

    def words(self, fileids=None):
        return self.corpus.words(fileids)

    def open(self, fileids):
        return self.corpus.open(fileids)

    def abspath(self, fileid):
        return self.corpus.abspath(fileid)

    # return list of all tf-idf vector
    # returns tf-idf corresponding to file
    # returns a list of vectors corresponding to the tf-idf of the file id inputs
    def tf_idfs(self, fileid=None, filelist=None):
        temp = []
        if fileid is not None:
            return list(self.weights[fileid].values())
        elif filelist is not None:
            for i in filelist:
                temp.append(list(self.weights[i].values()))
            return temp
        else:
            return self.weights
            for i in self.weights.values():
                temp.append(list(i.values()))
            return temp

    # returns the list of words in the order of the dimension of each corresponding to each vector of tf-idf vector
    def tf_idf_dim(self):
        one_file = list(self.weights.keys())[0]
        return list(self.weights[one_file].keys())

    # returns a vector corresponding to the tf_idf vector for the new document
    # words takes a list of words
    def tf_idf_new(self, words):
        new_document = {"new": words}
        self.__apply_considerations(new_document)
        for fileid in new_document:
            new_document[fileid] = dict(self.__term_frequency(new_document[fileid],self.tf))
            for keys in self.doc_idf:
                for files in list(new_document.values()):
                    if keys not in files:
                        files[keys] = 0
        return list(self.__calculate_weights(new_document)["new"].values())

    # returns the cosine similarity between two documents in the corpus
    # fileid is a list containing fileids of the two documents
    def cosine_sim(self, fileid):
        #try:
            if fileid[0] == fileid[1]:
                doc = {fileid[0]: self.weights[fileid[0]],"copy": self.weights[fileid[1]]}
                return self.__calculate_cosine_similarity(doc)
            else:
                doc = {fileid[0]: self.weights[fileid[0]],fileid[1]: self.weights[fileid[1]]}
                return self.__calculate_cosine_similarity(doc)
        #except IndexError:
           #return "No files to calculate cosine similarity"

    # returns the cosine similarity between a document and a new list in the corpus
    def cosine_sim_new(self, words, fileid):
        doc = {"newWord": dict(self.__term_frequency(words, self.tf))}
        doc = self.__apply_considerations(doc)
        doc = self.__calculate_weights(doc)
        doc[fileid] = self.weights[fileid]
        return self.__calculate_cosine_similarity(doc)

    # private methods
    def __apply_considerations(self, doc):
        # removing stop words
        if self.stopwords is not None:
            stp = set(stpwds.words(self.stopwords))
            for file in doc:
                doc[file] = [word for word in doc[file] if word not in stp]

        # stemming document
        if self.stemmer is not None:
            ps = self.stemmer
            for file in doc:
                doc[file] = [ps.stem(word) for word in doc[file]]
        return doc

#This function is used to calculate the term frequency for each file in the corpus
    def __term_frequency(self, doc, option="raw"):
        # ignoring case
        if self.ignorecase:
            file = FreqDist(map(str.lower, doc))
        else:
            file = FreqDist(doc)
        file = file.most_common()
        file = [list(word) for word in file]
        for index in range(len(file)):
                if option == "log":
                    file[index][1] = 1 + log2(file[index][1])
                elif option == "binary":
                    file[index][1] = 1
        return file

# This function is used to calculates the inverse document frequency for each file in the corpus
    def __inverse_document_frequency(self, i, number_of_docx, option="base"):
            if option == "smooth":
                x = log2(1 + (number_of_docx / i))
                return x
            elif option == "prob":
                try:
                    x = log2((number_of_docx - i) / i)
                    return  x
                except ValueError:
                    x = 0.0
                    return x
            else:
                x = log2(number_of_docx / i)
                return x


    def __calculate_weights(self, doc):
        w = {}
        for fileid in doc:
            tfidf = {}
            for word in doc[fileid]:
                tfidf[word] = doc[fileid][word] * self.doc_idf[word]
            w[fileid] = tfidf
        return w

    # type parameter is used to distinguish between if calculating for two file or a file and a new document
    # type 1 -> two files
    # type 2 -> a file and a document
    def __calculate_cosine_similarity(self,doc):
        keys =[]
        product = 1
        for i in doc:
            keys.append(i)
            if i is not "newWord":
                sum = 0
                for key,value in doc[i].items():
                    sum = sum + (value*value)
                sum = sqrt(sum)
                product =  product*sum
        dot_product = 0
        for key,values in doc[keys[0]].items():
            dot_product = dot_product + (values*doc[keys[1]][key])
        try:
            return dot_product/product
        except ZeroDivisionError:
            return "Undefined cosine similarity; division by zero"





