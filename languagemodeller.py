import collections
import itertools
import fileinput
import math

MIN_APPEARANCES = 5
UNKNOWN_WORD = "!!UNKNOWNWORD!!"
START_SYMBOL = "$$STARTSYMBOL$$"
STOP_SYMBOL = "^^STOPSYMBOL^^"

#combine multiple files
def combine_files(filenames):
    text = ''.join([line for line in fileinput.input(filenames)])
    return text

#turns string into list of constituent words
def string_to_words(s):
    return s.split(' ')

#replaces words with count less than MIN_APPEARANCES with "UNKNOWNWORD"
def kill_unknown_words(orig, text):
    killed = {}
    words = collections.Counter(text).most_common()
    for w, c in reversed(words):
        if c >= MIN_APPEARANCES:
            break
        killed[w] = True

    for i in xrange(len(text)):
        if text[i] in killed:
            text[i] = UNKNOWN_WORD
    
    return (text, killed)

#processes text, PUTS IN unknown shit
#run before anything else for all your preprocessing needs
def un_text(filenames):
    text = combine_files(filenames).lower()
    tokens = string_to_words(text)
    return kill_unknown_words(text, tokens)


#all the 3 functions below expect to be passed preprocessed data
#(UNKNOWN_WORD replacement done, processed into lists of words)

def unigrams(text):
    freqs = collections.Counter(text)
    freqs[STOP_SYMBOL] = 1
    return freqs


def bigrams(text):
    words = collections.Counter(text).most_common() #we need this later

    first = [START_SYMBOL] + text
    freqs = collections.Counter(itertools.izip(first, text))
    #now add (word, STOP) for every word in vocab
    for w, c in words:
        freqs[(w,STOP_SYMBOL)] = 1

    return freqs


def trigrams(text):
    words = collections.Counter(text).most_common() #we need this later

    first = [START_SYMBOL, START_SYMBOL] + text
    second = first[1:]
    freqs = collections.Counter(itertools.izip(first,second,text))
    #now add (word1, word2, STOP) for every pair of words in vocab
    for w1,w2 in itertools.combinations([w for w,c in words], 2):
        freqs[(w1,w2,STOP_SYMBOL)] = 1

    return freqs

#training_data = list of training data .txt files
#lambdas = list of coefficients in increasing subscript order
def generate_model(training_data, lambdas):
    (text,killed) = un_text(training_data)
    words = [w for w,c in collections.Counter(text).most_common()]
    vocab_size = len(words) + 1
    td_len = len(text) + 1
    freqs = (None, unigrams(text), bigrams(text), trigrams(text))

    #n = 0,1,2,3
    def mle(word, n, history):
        if n == 0:
            return 1/vocab_size #vocab_size's not going to be 0...

        lookup = history + tuple([word])
        if len(lookup) == 1:
            freq = freqs[n][word]
            all = td_len
        else:
            freq = freqs[n][lookup]
            all = freqs[n-1][history]

        if all == 0:
            return 0
        else:
            return float(freq)/float(all)

    def interpolate(word, history):
        p = 0
        for i in xrange(4):
            p += lambdas[i] * mle(word, i, history[:(i-1)])
        return p

    return (interpolate,killed,freqs[1])

#model is the function returned by generate_model()
#test_data is a filename
def perplexity(test_data, model, killed, unigrams):
    #using the log version
    s = 0.0
    count = 0
    text = [w.lower() for w in string_to_words(combine_files([test_data]))]

    for i in xrange(len(text)):
        if text[i] in killed or text[i] not in unigrams:
            text[i] = UNKNOWN_WORD


    text = [START_SYMBOL, START_SYMBOL] + text
    for i in xrange(2,len(text)):
        s += math.log(model(text[i], (text[i-2], text[i-1])))
        count += 1

    # print "s =", s, "count =", count
    return math.exp(-s/float(count))
