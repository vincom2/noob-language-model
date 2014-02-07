noob-language-model
===================

This is a quick and dirty python hack that, given some training data, generates a language model based off linear interpolation of
* a uniform model * L0
* unigram * L1
* bigram * L2
* trigram * L3
I'm putting it up mainly because I'm highly amused how generate_model() looks. I think I've been writing too much SML.

##Usage
$ lm.py L0 L1 L2 L3 test.txt train.txt<br/>
where the L's are the lambda coefficients for the uniform model up to the trigram model.<br/>
<br/>

###Todo
* generalise to higher-N models?