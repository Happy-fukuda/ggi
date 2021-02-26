from collections import defaultdict
import gensim.downloader as Word2Vec
from sklearn.cluster import KMeans

model = Word2Vec.load('glove-twitter-200')
max_vocab = 30000
vocab = list(model.wv.vocab.keys())[:max_vocab]
vectors = [model.wv[word] for word in vocab]
n_clusters = 1000
kmeans_model = KMeans(n_clusters=n_clusters, verbose=1, random_state=42, n_jobs=-1)
kmeans_model.fit(vectors)
print("finish fit")
cluster_labels = kmeans_model.labels_
cluster_to_words = defaultdict(list)
for cluster_id, word in zip(cluster_labels, vocab):
    cluster_to_words[cluster_id].append(word)

for words in cluster_to_words.values():
    print(words[:10])
