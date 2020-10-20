from nltk.corpus import brown,reuters,gutenberg
import json
from string import  punctuation


def sentences_from_corpus():
    corpusfiles = brown.fileids()
    corpusSize = len(corpusfiles)

    temp = []
    tokens_sentences = []
    for i in range(corpusSize):
        temp.append(brown.sents(corpusfiles[i]))
        for j in range(len(temp[i])):
            tokens_sentences.append(temp[i][j])
    return tokens_sentences


def createkey(x):
    key = ""
    for i in x:
        if key == "":
            key = i.lower().strip()
        else:
            key = key+"-"+i.lower().strip()
    return key


def calculateProbability(file,totals,freq):
    ngram_prob_dict = {}
    for keys in totals.keys():
        sample_space = totals[keys]

        for events in freq[keys].keys():
            event = freq[keys][events]
            if keys in ngram_prob_dict.keys():
                ngram_prob_dict[keys][events] = event/sample_space
            else:
                ngram_prob_dict[keys] = {events: event / sample_space}

    with open(file,'w') as f:
        for chunk in json.JSONEncoder().iterencode(ngram_prob_dict):
            f.write(chunk)
    return


def ngrams(file,corpus, ngram):
    ngram_total_dict = {}
    ngram_freq_dict = {}

    for words in corpus:
        words = [''.join(c for c in s if c not in punctuation) for s in words]
        words = [s for s in words if s]
        word_len = len(words)
        for i in range(word_len):
            start = i+1
            if i == 0 or i== word_len-1:
                start = i
            end = i+ngram+1
            if end > word_len-1:
                end = word_len
            for j in range(start,end,1):
                prev = createkey(words[i:j])
                current = words[j].lower()
                key = prev+"___"+current
                if not prev :
                    if  i==0:
                        prev = "__"
                    if i== word_len-1:
                        prev = current
                        current = "__"
                        key = prev + current
                if prev in ngram_total_dict:
                    ngram_total_dict[prev] = ngram_total_dict[prev]+1
                else:
                    ngram_total_dict[prev] = 1
                if prev in ngram_freq_dict:
                    if current in ngram_freq_dict[prev]:
                        ngram_freq_dict[prev][current] = ngram_freq_dict[prev][current]+1
                    else:
                        ngram_freq_dict[prev][current] = 1
                else:
                    ngram_freq_dict[prev] = {current:1}
    return calculateProbability(file,ngram_total_dict, ngram_freq_dict)


ngrams("brown.txt",sentences_from_corpus(),ngram=3)
