import scipy as sp
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk.stem


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        english_stemmer = nltk.stem.SnowballStemmer('english')
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def tfidf(term, doc, corpus):
    tf = doc.count(term) / len(doc)
    num_docs_with_term = len([d for d in corpus if term in d])
    idf = sp.log(len(corpus) / num_docs_with_term)
    return tf * idf


groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
          'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.fetch_20newsgroups(subset='train', categories=groups)
# print(len(train_data.filenames))
test_data = sklearn.datasets.fetch_20newsgroups(subset='test', categories=groups)
# print(len(test_data.filenames))
vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape
# print("#isamples: %d, features: %d" % (num_samples, num_features))

# cluster qty
num_clusters = 50
# del random_state in real app
km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1, random_state=3)
km.fit(vectorized)
# print(km.labels_)               # post labels
# print(km.cluster_centers_)      # cluster centroids

# test with sample post:
new_post = 'Disk drive problems. Hi, i have a problem with my hard disk. ' \
           'After 1 year it is working only sporadically now.' \
            'I tried to format it, but now it doesnt boot any more. Any ideas? Thanks.'
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
similar_indices = (km.labels_ == new_post_label).nonzero()[0]
similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))
    similar = sorted(similar)
print(len(similar))
print (similar[0], similar[50])






