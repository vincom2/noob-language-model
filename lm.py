#!/usr/bin/python

import languagemodeller as lm
import sys

def main():
    if len(sys.argv) !=7:
        print "lm.py: generates a linearly interpolated language model with n from 0 to 3 from train.txt"
        print "and computes its perplexity score on test.txt."
        print "usage: lm.py <L0> <L1> <L2> <L3> test.txt train.txt"
        sys.exit(1)

    lambdas = []
    for i in xrange(1, 5):
        lambdas.append(float(sys.argv[i]))

    training_data = sys.argv[6]
    test_data = sys.argv[5]

    model = lm.generate_model(training_data, lambdas)
    print "With test data", test_data, "perplexity is", lm.perplexity(test_data, *model)

if __name__ == '__main__':
    main()
