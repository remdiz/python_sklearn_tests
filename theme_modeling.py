# import matplotlib.pyplot as plt
from scipy.spatial import distance
from gensim import corpora, models, matutils

# Associated Press news data http://www.cs.princeton.edu/~blei/lda-c/ap.tgz
corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')

# theme modeling
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word)
doc = corpus.docbyoffset(0)
topics = model[doc]
# document associated topics (topic index/weight pairs)
# print(topics)

# theme matrix
topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
# pair distance
pairwise = distance.squareform(distance.pdist(topics))
# set max value to diagonal matrix elements
largest = pairwise.max()
for ti in range(len (topics)):
    pairwise[ti, ti] = largest + 1


# find closest doc (closest neighbor classification)
def closest_to(doc_id):
    return pairwise[doc_id].argmin()

