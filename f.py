
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def read_files(file_list):
    corpus = []
    for file_path in file_list:
        with open(file_path) as f_input:
            corpus.append(f_input.read())
    return corpus

def cluster_documents(corpus, num_clusters):
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

   
    agglomerative_clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
    cluster_labels = agglomerative_clustering.fit_predict(cosine_similarity(tfidf_matrix))
  # Print cluster labels for each document
    print("\nCLUSTER LABELS FOR EACH DOCUMENT:")
    for i, label in enumerate(cluster_labels):
        print(f"Document {i + 1}: Cluster {label}")

    return cluster_labels

def predict_cluster(test_file, corpus, vectorizer, clustering_model):
    file
    with open(test_file) as f:
        test_content = f.read()

    # Add the test document to the corpus
    corpus.append(test_content)

    # Transform the corpus using the existing vectorizer
    tfidf_matrix = vectorizer.transform(corpus)

    # Predict the cluster of the test document
    test_cluster = clustering_model.fit_predict(cosine_similarity(tfidf_matrix))[-1]

    print("\nALGORITHM HAS PREDICTED THAT THE GIVEN TEST FILE BELONGS TO CLUSTER NO.:", test_cluster)

# Set paths for clustering and test files
clustering_files = glob.glob(os.path.join(os.getcwd(), "clustering", "*.txt"))
test_file_path = glob.glob(os.path.join(os.getcwd(), "test", "abcc.txt"))[0]

# Read the corpus from the clustering files
corpus = read_files(clustering_files)

# Number of clusters
num_clusters = 5

# Cluster the documents
cluster_labels = cluster_documents(corpus, num_clusters)

# Test the algorithm on a new file
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
predict_cluster(test_file_path, corpus, vectorizer, AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average'))
