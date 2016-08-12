import matplotlib.pyplot as plt
from gensim import corpora, models

# Associated Press news data http://www.cs.princeton.edu/~blei/lda-c/ap.tgz
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

# theme modeling
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)
doc = corpus.docbyoffset(0)
topics = model[doc]
# document associated topics (topic index/weight pairs)
# print(topics)
